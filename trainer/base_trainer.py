import os
import torch
import models
import numpy as np
import os.path as osp
import torch.optim as optim
import dataloader as enose_dataset
from torch.utils.data import DataLoader
from losses.base import TrainingLoss
from utils.common import get_valid_args, ts2np
from utils.utils import cal_metrics, get_confusion_matrix, plot_confusion_matrix, plot_features, plot_loss

class BaseTrainer():
    def __init__(self, cfgs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        self.trainer_cfg = cfgs['trainer_cfg']

        self.train_loader, self.test_loader = self.build_loaders(cfgs['data_cfg'], cfgs['dataloader_cfg'])
        
        self.model = self.build_model(cfgs['model_cfg']).to(self.device)
        self.model_name = cfgs['model_cfg']['model_name']  
        
        self.criterion = TrainingLoss(cfgs['loss_cfg'])
        self.optimizer = self.get_optimizer(cfgs['optimizer_cfg'])
        self.lr_scheduler = self.get_scheduler(cfgs['scheduler_cfg'])

        self.start_epoch = 0
        self.best_test_metric = self.trainer_cfg['test_metric_threshold']
       
        self.epochs = self.trainer_cfg['epochs']
        self.test_metric_threshold = self.trainer_cfg['test_metric_threshold']
        self.early_stop = self.trainer_cfg['early_stop'] 

        self.trainer_name = self.trainer_cfg['type']
        self.target_class_idx = self.trainer_cfg['target_class_idx']

        self.save_dir = osp.join(self.trainer_cfg['save_dir'], self.model_name, self.trainer_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        restore_hint = self.trainer_cfg['restore_hint']
        self.load(restore_hint)
        
        if 'pretrained_path' in self.trainer_cfg:
            self.load_pretrained(self.trainer_cfg['pretrained_path'])

    def build_loaders(self, data_cfg, loader_cfg):
        Dataset = getattr(enose_dataset, data_cfg['type'])
        valid_dataset_args = get_valid_args(Dataset, data_cfg)
        train_dataset = Dataset(flag='train', **valid_dataset_args)
        test_dataset = Dataset(flag='test', **valid_dataset_args)

        
        valid_loader_args = get_valid_args(DataLoader, loader_cfg)
        train_loader = DataLoader(train_dataset, shuffle=True, **valid_loader_args)
        test_loader = DataLoader(test_dataset, shuffle=False, **valid_loader_args)

        return train_loader, test_loader

    def build_model(self, model_cfg):
        Model = getattr(models, model_cfg['type'])
        valid_model_args = get_valid_args(Model, model_cfg, ['type', 'model_name'])
        model = Model(**valid_model_args)
        return model
    
    def get_optimizer(self, optimizer_cfg):
        Optimizer = getattr(optim, optimizer_cfg['type'])
        valid_arg = get_valid_args(Optimizer, optimizer_cfg, ['type'])
        model_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        loss_params = list(filter(lambda p: p.requires_grad, self.criterion.parameters()))
        params = model_params + loss_params
        optimizer = Optimizer(params, **valid_arg)
        return optimizer

    def get_scheduler(self, scheduler_cfg):
        Scheduler = getattr(optim.lr_scheduler, scheduler_cfg['type'])
        valid_arg = get_valid_args(Scheduler, scheduler_cfg, ['type'])
        scheduler = Scheduler(self.optimizer, **valid_arg)
        return scheduler

    def train(self):
        self.model.train()
        train_loss_list = []
        test_loss_list = []
        no_improvement = 0
        for i in range(self.start_epoch+1, self.epochs+1):
            print(f'############ Epoch: {i} start ###############')
            batch_train_loss = []
            preds = []
            gts = []
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                logits, feats = self.model(data)
                loss, loss_info = self.criterion(logits, feats, target)
                main_logits = logits[0] if isinstance(logits, list) else logits
                pred = main_logits.argmax(dim=1, keepdim=True)
                preds.append(ts2np(pred))
                gts.append(ts2np(target))
                loss.backward()
                self.optimizer.step()
                batch_train_loss.append(loss.item())

            self.lr_scheduler.step()

            train_loss = np.average(batch_train_loss)
            preds, gts = np.concatenate(preds), np.concatenate(gts)
            train_metrics = cal_metrics(gts, preds, target_cls_id=self.target_class_idx, stage='Training')
            test_loss, test_metrics = self.test(model_epoch=i, is_testing=False)
            test_average = test_metrics['average']
            
            if test_average > self.best_test_metric:
                self.best_test_metric = test_average
                self.save(i, test_average, test_best=True)
                no_improvement = 0
            else:
                no_improvement += 1

            print(f'Epoch: {i}, Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
            for key, value in train_metrics.items():
                print(f'Training {key}: {value:.4f}, Test {key}: {test_metrics[key]:.4f}')

            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            
            print()
            
            if no_improvement >= self.early_stop:
                print(f'Early stopping at epoch {i}')
                break
        
        plot_loss(train_loss_list, test_loss_list, self.save_dir)

        self.load(epoch=0)
        self.test(is_testing=True)

    def test(self, model_epoch=None, is_testing=True):
        self.model.eval()
        total_loss = []
        preds = []
        gts = []
        feats_list = []
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                logits, feats = self.model(data)
                loss, loss_info = self.criterion(logits, feats, target)
                main_logits = logits[0] if isinstance(logits, list) else logits 
                pred = main_logits.argmax(dim=1, keepdim=True)
                feats_list.append(ts2np(feats))
                preds.append(ts2np(pred))
                gts.append(ts2np(target))
                total_loss.append(loss.item())
        
        total_loss = np.average(total_loss)
        preds, gts = np.concatenate(preds), np.concatenate(gts)
        metrics = cal_metrics(gts, preds, target_cls_id=self.target_class_idx, stage='Test')

        vis_threshold = self.test_metric_threshold

        if metrics['average'] > vis_threshold or is_testing:
            model_epoch = self.start_epoch if model_epoch is None else model_epoch
            for key, value in metrics.items():
                print(f'Test {key} on epoch {model_epoch}: {value:.4f}')
            confusion_matrix = get_confusion_matrix(gts, preds)
            plot_confusion_matrix(confusion_matrix, self.save_dir, save_name=f'epoch_{model_epoch}_test_confusion_matrix.jpg')
            feats_list = np.concatenate(feats_list, axis=0)
            plot_features(feats_list, gts, self.save_dir, save_name=f'epoch_{model_epoch}_test_features.jpg')
        
        self.model.train()
        return total_loss, metrics
    
    
    def save(self, epoch, test_average, test_best=False):
        ckpt = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'test_average': test_average,
        }
        print(f'Saving model at epoch {epoch} with test average metric: {test_average:.4f} on test set')
        if test_best:  
            torch.save(ckpt, f'{self.save_dir}/best_test_model.pt')
        torch.save(ckpt, f'{self.save_dir}/model_epoch_{epoch}.pt')
        

    def load(self, epoch=0):
        if epoch == 0:
            path = os.path.join(self.save_dir, 'best_test_model.pt')
        else:
            path = os.path.join(self.save_dir, f'model_epoch_{epoch}.pt')
        if os.path.exists(path):
            ckpt = torch.load(path)
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
            self.start_epoch = ckpt['epoch']
            self.best_test_metric = ckpt['test_average']
            print(f"Resuming checkponit from epoch: {self.start_epoch}")
        else:
            print(f'No checkpoint found at {path}, starting from scratch')
    
    def load_pretrained(self, path):
        ckpt = torch.load(path)
        ckpt = ckpt['model']
        # remove linear layer
        new_dict = {}
        for k, v in ckpt.items():
            if 'linear' not in k:
                new_dict[k] = v
        self.model.load_state_dict(new_dict, strict=False)
        print(f'Pretrained model loaded from {path}')


