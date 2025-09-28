import os
import sys
sys.path.append('./')
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from dataloader.dataset import EnoseDataset
from utils.common import config_loader, get_valid_args
from utils.utils import get_confusion_matrix, cal_metrics, plot_confusion_matrix

config_path = './configs/base.yaml'
save_dir = "./outputs/knn"

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

# Train the KNN model
model = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
model.fit(train_data, train_label)

# Test the model
preds = model.predict(test_data)

confusion_matrix = get_confusion_matrix(test_label, preds)
plot_confusion_matrix(confusion_matrix, save_dir)
test_metrics = cal_metrics(test_label, preds, target_cls_id=0, stage='Test')
for key, value in test_metrics.items():
    print(f"Test {key}: {value:.4f}")

