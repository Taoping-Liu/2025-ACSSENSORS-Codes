import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # set the GPU to use
import trainer as trainers
import argparse
from utils.common import config_loader

def init_seed(seed=0):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser(description='Main program for UBC screening.')
parser.add_argument('--cfgs', type=str,
                    default='configs/base.yaml', help="path of config file")
parser.add_argument('--phase', default='train',
                    choices=['train', 'test'], help="choose train / test phase")
parser.add_argument('--iter', type=int, default=0, help="iter to restore")
parser.add_argument('--seed', type=int, default=0, help="random seed")
opt = parser.parse_args()

def main(cfgs, phase):
    Trainer = getattr(trainers, cfgs['trainer_cfg']['type'])
    trainer = Trainer(cfgs)
    if phase == 'train':
        trainer.train()
    elif phase == 'test':
        trainer.test()

if __name__ == '__main__':
    cfgs = config_loader(opt.cfgs)
    seed = cfgs['data_cfg']['seed'] if 'seed' in cfgs['data_cfg'] else opt.seed
    init_seed(seed)
    if opt.iter > 0:
        cfgs['trainer_cfg']['restore_hint'] = int(opt.iter)
    main(cfgs, opt.phase)