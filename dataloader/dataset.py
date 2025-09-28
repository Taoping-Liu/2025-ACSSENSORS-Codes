import os
import sys
sys.path.append('./')
import torch
import random
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.common import is_list, is_ndarray
from utils.utils import plot_features
import dataloader.tsgm_augmentations as tsgm_augs


def scale_data(data, scaler='Minmax'):
    scaler = StandardScaler() if scaler == 'Standard' else MinMaxScaler()
    data = scaler.fit_transform(data)
    return data

class EnoseDataset(Dataset):
    all_data = None 
    label_mapping = None 

    def __init__(self, data_dir, dataset_cfg=None, flag='train', seed=0, scale=True):
        self.data_dir = data_dir
        self.dataset_cfg = dataset_cfg
        self.channels = dataset_cfg['channels']
        self.flag = flag
        self.seed = seed
        if EnoseDataset.all_data is None:
            EnoseDataset.all_data = EnoseDataset.load_and_split_data(data_dir, dataset_cfg, seed, scale)

        self.data_list, self.label_list = EnoseDataset.get_dataset_by_flag(flag)
        
        if flag == 'train' and self.dataset_cfg['use_tsgm']:
            self.generate_data_by_tsgm()

        assert len(self.data_list) == len(self.label_list), 'The length of data list and label list should be the same!'

    def generate_data_by_tsgm(self):
        cfg_list = self.dataset_cfg['tsgm_cfg']
        save_dir = self.dataset_cfg['tsgm_save_dir']
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        all_new_data, all_new_label = [], []
        for cfg in cfg_list:
            aug_type = cfg['type']
            Aug_model = getattr(tsgm_augs, aug_type)
            aug_model = Aug_model()
            class_id_list = cfg['class_id_list'] 
            gen_num_list = cfg['gen_num_list'] 
            gen_params = cfg['gen_params']
            gen_params = {} if gen_params is None else gen_params
            assert len(class_id_list) == len(gen_num_list), 'The length of class_id_list and gen_num_list should be the same!'
            for i in range(len(class_id_list)):
                class_id = class_id_list[i]
                gen_num = gen_num_list[i]
                data_path = os.path.join(save_dir, f'{class_id}_{gen_num}_{aug_type}_data.npy')
                label_path = os.path.join(save_dir, f'{class_id}_{gen_num}_{aug_type}_label.npy')
                if os.path.exists(data_path) and os.path.exists(label_path):
                    new_data = np.load(data_path)
                    new_label = np.load(label_path) 
                    if len(new_label.shape) == 2:
                        new_label = new_label.reshape(-1)
                else:
                    source_data, source_label = self.get_data_by_labels([class_id])
                    print(f'Generate {gen_num} samples with label {class_id}')
                    new_data, new_label = aug_model.generate(X=source_data, y=source_label, 
                                                                    n_samples=gen_num, **gen_params)
                    if len(new_label.shape) == 2:
                        new_label = new_label.reshape(-1)
                    np.save(data_path, new_data)
                    np.save(label_path, new_label)
                all_new_data.append(new_data)
                all_new_label.append(new_label)
        all_new_data = np.concatenate(all_new_data, axis=0)
        all_new_label = np.concatenate(all_new_label, axis=0)
        self.vis_tsgm_data()
        self.add_samples(all_new_data, all_new_label)
        print(f'Add {len(all_new_data)} samples to train dataset!')
        
    def vis_tsgm_data(self):
        save_dir = self.dataset_cfg['tsgm_save_dir']
        file_list = os.listdir(save_dir)
        aug_class_list = []
        agu_info_list = []
        for file_name in file_list:
            if 'data.npy' in file_name:
                class_id = int(file_name.split('_')[0])
                aug_num = int(file_name.split('_')[1])
                aug_type = file_name.split('_')[2]
                if class_id not in aug_class_list:
                    aug_class_list.append(class_id)
                agu_info_list.append((class_id, aug_num, aug_type))
        agu_label_mapping = {}
        for class_id in aug_class_list:
            class_ori_data, class_ori_label = self.get_data_by_labels([class_id])
            class_ori_label = class_ori_label * 0
            class_gen_data, class_gen_label = [], []
            for aug_id, aug_num, aug_type in agu_info_list:
                if class_id == aug_id:
                    data_path = os.path.join(save_dir, f'{class_id}_{aug_num}_{aug_type}_data.npy')  
                    new_data = np.load(data_path)
                    if aug_type not in agu_label_mapping.keys():
                        agu_label_mapping[aug_type] = len(agu_label_mapping) + 1
                    new_label = np.ones(new_data.shape[0]) * agu_label_mapping[aug_type]
                    class_gen_data.append(new_data)
                    class_gen_label.append(new_label)
            class_gen_data = np.concatenate(class_gen_data, axis=0)
            class_gen_label = np.concatenate(class_gen_label, axis=0)
            class_data = np.concatenate([class_ori_data, class_gen_data], axis=0)
            class_label = np.concatenate([class_ori_label, class_gen_label], axis=0)
            # flatten data
            class_data = class_data.reshape(class_data.shape[0], -1)
            class_label = class_label.astype(np.int32)
            plot_features(class_data, class_label, save_dir, save_name=f'{class_id}_tsgm.jpg', save_features=False)
                
                                

    @staticmethod
    def get_data_by_labels(labels):
        train_data_list, train_label_list = EnoseDataset.all_data[0]
        label_mapping = EnoseDataset.label_mapping
        mapped_labels = [label_mapping[label] for label in labels]
        data_list, label_list = [], []
        for data, label in zip(train_data_list, train_label_list):
            if label in mapped_labels:
                data_list.append(data)
                idx = mapped_labels.index(label)
                org_label = labels[idx] 
                label_list.append(org_label)
        return np.array(data_list), np.array(label_list)

    @staticmethod
    def get_label_mapping(label_list):
        unique_labels = sorted(set(label_list))
        label_mapping = {label: i for i, label in enumerate(unique_labels)}
        return label_mapping

    @staticmethod
    def get_dataset_by_flag(flag):
        if flag == 'train':
            data_list, label_list = EnoseDataset.all_data[0]
        elif flag == 'test':
            data_list, label_list = EnoseDataset.all_data[1]
        else:
            raise ValueError('Invalid flag! Please use one of the following: train/test, but got {}'.format(flag))
        return data_list, label_list

    @staticmethod
    def load_and_split_data(data_dir, data_cfg, seed=0, scale=True):
        data_list, label_list = EnoseDataset.load_all_data(data_dir, data_cfg, scale)
        class_train_samples = data_cfg['class_train_samples'] 
        data_array = np.array(data_list)
        label_array = np.array(label_list)

        train_data_list, test_data_list = [], []
        train_label_list, test_label_list = [], []

        unique_labels = np.unique(label_array)

        for label in unique_labels:
            indices = np.where(label_array == label)[0]

            np.random.seed(seed)

            np.random.shuffle(indices)

            num_samples = len(indices)
            train_size = int(0.8 * num_samples)  # 80% for training

            if label in class_train_samples.keys():
                train_size = class_train_samples[label]
            
            train_indices = indices[:train_size]
            test_indices = indices[train_size:]
            
            train_data_list.extend(data_array[train_indices])
            test_data_list.extend(data_array[test_indices])

            mapping_label = EnoseDataset.label_mapping[label]

            
            train_label_list.extend([mapping_label] * len(train_indices)) 
            test_label_list.extend([mapping_label] * len(test_indices))

        return (train_data_list, train_label_list), (test_data_list, test_label_list)

    @staticmethod
    def load_all_data(data_dir, data_cfg, scale=True):
        seq_len = data_cfg['seq_len'] 
        all_data_list = []
        all_label_list = []
        all_class_list = data_cfg['All']
        for class_name in all_class_list:
            class_dir = os.path.join(data_dir, class_name)
            file_name_list = data_cfg[class_name]
            for file_name in file_name_list:
                file_path = os.path.join(class_dir, file_name+'.mat')
                data = sio.loadmat(file_path)[file_name] # [N_sample, 1]
                label = int(file_name.split('_')[-1])
                for i in range(data.shape[0]):
                    sample_data = data[i][0][:seq_len].astype(np.float32) # [seq_len, num_channels]
                    sample_data = np.nan_to_num(sample_data)
                    if scale:
                        sample_data = scale_data(sample_data)
                    all_data_list.append(sample_data)
                    all_label_list.append(label)
        EnoseDataset.label_mapping = EnoseDataset.get_label_mapping(all_label_list)
        return all_data_list, all_label_list

    
    def add_samples(self, datas, labels):
        if self.flag != 'train':
            raise ValueError('Only train dataset can add samples!')
        label_mapping = EnoseDataset.label_mapping
        if is_list(datas):
            self.data_list.extend(datas)
            labels = [label_mapping[label] for label in labels]
            self.label_list.extend(labels)
        elif is_ndarray(datas) :
            if len(datas.shape) == 2: # [seq_len, num_channels]
                self.data_list.append(datas)
                labels = label_mapping[labels]
                self.label_list.append(labels)
            elif len(datas.shape) == 3: # [N_samples, seq_len, num_channels]
                self.data_list.extend(datas)
                labels = labels.reshape(-1) # [N_samples]
                labels = [label_mapping[label] for label in labels]
                self.label_list.extend(labels)
        else:
            raise ValueError('Invalid type of datas! Should be list or np.ndarray! Got {}'.format(type(datas)))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index] # [seq_len, num_channels]
        data = data[:,self.channels]
        label = self.label_list[index]
        return data.astype(np.float32), torch.tensor(label, dtype=torch.long)
       
 
if __name__ == '__main__':
    from utils.common import config_loader, get_valid_args
       
    config_path = './configs/base.yaml'
    
    cfg = config_loader(config_path)
    data_cfg = cfg['data_cfg']
    data_cfg['dataset_cfg']['All'] = ['UBC', 'Healthy', 'Water', 'VOC']
    data_cfg['dataset_cfg']['use_tsgm'] = True
    data_scale = True
    data_cfg['scale']= data_scale
    valid_dataset_args = get_valid_args(EnoseDataset, data_cfg)
    train_dataset = EnoseDataset(flag='train', **valid_dataset_args)
    selected_classes = [0, 1, 4, 7, 10, 11, 12, 13]
    print('training set')
    train_labels = train_dataset.label_list
    label_unique = np.unique(train_labels, return_counts=True)
    # show count of selected classes
    for index, class_i in enumerate(selected_classes):
        print(f'Class {class_i} count: {label_unique[1][index]}')
    test_dataset = EnoseDataset(flag='test', **valid_dataset_args)
    print('testing set')
    test_labels = test_dataset.label_list
    label_unique = np.unique(test_labels, return_counts=True)
    for index, class_i in enumerate(selected_classes):
        print(f'Class {class_i} count: {label_unique[1][index]}')
        