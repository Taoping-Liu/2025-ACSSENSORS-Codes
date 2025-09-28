import os
import sys
sys.path.append('./')
import numpy as np
from sklearn.mixture import GaussianMixture
from dataloader.dataset import EnoseDataset
from utils.common import config_loader, get_valid_args
from utils.utils import get_confusion_matrix, cal_metrics, plot_confusion_matrix

config_path = './configs/base.yaml'
save_dir = "./outputs/gmm_per_class"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

cfg = config_loader(config_path)
data_cfg = cfg['data_cfg']

valid_dataset_args = get_valid_args(EnoseDataset, data_cfg)
train_dataset = EnoseDataset(flag='train', **valid_dataset_args)
test_dataset = EnoseDataset(flag='test', **valid_dataset_args)

train_data, train_label = train_dataset.data_list, train_dataset.label_list
test_data, test_label = test_dataset.data_list, test_dataset.label_list

# Convert to numpy array and flatten the data
train_data = np.array(train_data).reshape(len(train_data), -1)
train_label = np.array(train_label)
test_data = np.array(test_data).reshape(len(test_data), -1)
test_label = np.array(test_label)

# Get unique classes
classes = np.unique(train_label)

# Train a GMM for each class
gmms = {}
for cls in classes:
    cls_data = train_data[train_label == cls]
    gmm = GaussianMixture(n_components=1, random_state=42)
    gmm.fit(cls_data)
    gmms[cls] = gmm

# Predict the class of each sample in the test set
preds = []
for sample in test_data:
    log_probs = []
    for cls in classes:
        log_prob = gmms[cls].score_samples(sample.reshape(1, -1))
        log_probs.append(log_prob)
    preds.append(classes[np.argmax(log_probs)])

# Convert predictions to numpy array
preds = np.array(preds)

# Evaluate the model
confusion_matrix = get_confusion_matrix(test_label, preds)
plot_confusion_matrix(confusion_matrix, save_dir)
test_metrics = cal_metrics(test_label, preds, target_cls_id=0, stage='Test')
for key, value in test_metrics.items():
    print(f"Test {key}: {value:.4f}")
