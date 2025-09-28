from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os.path as osp
import numpy as np
from sklearn.manifold import TSNE

def plot_features(feats, labels, save_dir, save_name='features.jpg', save_features = True):
    """
    Args:
        feats (np.ndarray): shape: [num_samples, num_features]
        labels (np.ndarray): shape: [num_samples]
    """
    plt.clf()

    n_samples = feats.shape[0]

    perplexity_value = min(10, n_samples - 1)

    tsne = TSNE(perplexity=perplexity_value, n_components=2)
    feats_embedded = tsne.fit_transform(feats)
    class_names = [f'class_{i}' for i in np.unique(labels)]
    if save_features:
        file_name = save_name.split('.')[0]
        feats_name = file_name + '_feats.npy'
        labels_name = file_name + '_labels.npy'
        np.save(osp.join(save_dir, feats_name), feats_embedded)
        np.save(osp.join(save_dir, labels_name), labels)

    scatter = plt.scatter(feats_embedded[:, 0], feats_embedded[:, 1], c=labels,  cmap='tab20')
    plt.legend(handles=scatter.legend_elements()[0], labels=class_names, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout() 
    plt.savefig(osp.join(save_dir, save_name))


def cal_metrics(gts, preds, target_cls_id=None, stage='Test'):
    assert stage in ['Training', 'Test'], f'stage should be one of [Training, Test], but got {stage}'
    if target_cls_id is not None:
        target_gt = np.where(gts == target_cls_id, 1, 0)
        target_pred = np.where(preds == target_cls_id, 1, 0)
        target_f1 = f1_score(target_gt, target_pred)
        print(f'{stage} f1 for class {target_cls_id}: {target_f1}')
    accuracy = accuracy_score(gts, preds)
    avg_type = 'macro'
    precision = precision_score(gts, preds, average=avg_type, zero_division=np.nan) # add zero_division to avoid warning
    recall = recall_score(gts, preds, average=avg_type, zero_division=np.nan)
    f1 = f1_score(gts, preds, average=avg_type, zero_division=np.nan)
    average = (accuracy + precision + recall + f1) / 4
    dict_metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'average': average}
    return dict_metrics

def get_confusion_matrix(trues, preds):
    conf_matrix = confusion_matrix(trues, preds)
    return conf_matrix

def plot_confusion_matrix(conf_matrix, save_dir, save_name='confusion_matrix.jpg'):
    plt.clf()
    plt.imshow(conf_matrix, cmap=plt.cm.Greens)
    indices = range(conf_matrix.shape[0])
    labels = []
    for i in range(len(conf_matrix)):
        labels.append(i)

    plt.xticks(indices, labels)
    plt.yticks(indices, labels)
    plt.colorbar()
    plt.xlabel('y_true')
    plt.ylabel('y_pred')
    # plot data
    for first_index in range(conf_matrix.shape[0]):
        for second_index in range(conf_matrix.shape[1]):
            plt.text(first_index, second_index, conf_matrix[first_index, second_index])
    plt.savefig(osp.join(save_dir, save_name))


def plot_loss(train_loss_list, test_loss_list, save_dir):
    plt.clf()
    plt.plot(train_loss_list, label='train_loss')
    plt.plot(test_loss_list, label='test_loss')
    plt.legend()
    plt.savefig(osp.join(save_dir, 'loss.jpg'))

