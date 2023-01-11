#%%
import torch
import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np


def test(model, data_loader, task_num, device, step_callbacks=None):
    model.eval()
    labels_dict, predicts_dict, loss_dict = {}, {}, {}
    for i in range(task_num):
        labels_dict[i], predicts_dict[i], loss_dict[i] = list(), list(), list()
    with torch.no_grad():
        for j, (categorical_fields, numerical_fields, labels) in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)):
            categorical_fields, numerical_fields, labels = categorical_fields.to(device), numerical_fields.to(device), labels.to(device)
            y = model(categorical_fields, numerical_fields)
            for i in range(task_num):
                labels_dict[i].extend(labels[:, i].tolist())
                predicts_dict[i].extend(y[i].tolist())
                loss_dict[i].extend(torch.nn.functional.binary_cross_entropy(y[i], labels[:, i].float(), reduction='none').tolist())
            stop_iteration = False
            for callback in step_callbacks or []:
                try:
                    callback(step=j)
                except StopIteration:
                    stop_iteration = True
            if stop_iteration:
                break
    auc_results, loss_results = list(), list()
    for i in range(task_num):
        auc_results.append(roc_auc_score(labels_dict[i], predicts_dict[i]))
        loss_results.append(np.array(loss_dict[i]))
    return auc_results, loss_results
# %%
