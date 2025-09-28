import os
import sys
sys.path.append('./')
import numpy as np
from sklearn.svm import SVC
from dataloader.dataset import EnoseDataset
from utils.common import config_loader, get_valid_args
from utils.utils import get_confusion_matrix, cal_metrics, plot_confusion_matrix

config_path = './configs/new.yaml'
save_dir = "./outputs/svm_stage"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

cfg = config_loader(config_path)
data_cfg = cfg['data_cfg']

valid_dataset_args = get_valid_args(EnoseDataset, data_cfg)
train_dataset = EnoseDataset(flag='train', **valid_dataset_args)
test_dataset = EnoseDataset(flag='test', **valid_dataset_args)

train_data, train_label = train_dataset.data_list, train_dataset.label_list
test_data, test_label = test_dataset.data_list, test_dataset.label_list

# convert to numpy array
train_data = np.array(train_data) # [num_samples, length, channel]
# flatten the data
train_data = train_data.reshape(train_data.shape[0], -1) # [num_samples, length*channel]
train_label = np.array(train_label) # [num_samples]


test_data = np.array(test_data) # [num_samples, length, channel]
# flatten the data
test_data = test_data.reshape(test_data.shape[0], -1) # [num_samples, length*channel]
test_label = np.array(test_label) # [num_samples]

# train the svm model
model = SVC(kernel='rbf', C=1, gamma='auto', random_state=42)
model.fit(train_data, train_label)

# test the model
preds = model.predict(test_data) # [num_samples]

confusion_matrix = get_confusion_matrix(test_label, preds)
plot_confusion_matrix(confusion_matrix, save_dir)
test_metrics = cal_metrics(test_label, preds, target_cls_id=0, stage='Test')
for key, value in test_metrics.items():
    print(f"Test {key}: {value:.4f}")

    