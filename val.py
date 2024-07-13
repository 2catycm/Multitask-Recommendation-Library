#%%
import torch
import tqdm
from sklearn.metrics import roc_auc_score, r2_score
import numpy as np
from tensorboardX import SummaryWriter
tensorboard = SummaryWriter('./tensorboard_log')

# 我对使用的模型进行了讨论，代码的正确性不能保证 //from oyl
def test(model, data_loader, task_num, device, epoch=0, step_callbacks=None, loss_type='MSELoss'.lower()):
    if loss_type == 'BCELoss'.lower() or loss_type == 'BCEWithLogitsLoss'.lower() or loss_type == 'AUCMLoss'.lower():
        model.eval()
        labels_dict, predicts_dict, loss_dict = {}, {}, {}
        for i in range(task_num):
            labels_dict[i], predicts_dict[i], loss_dict[i] = list(), list(), list()
        with torch.no_grad():
            for j, (categorical_fields, numerical_fields, labels) in enumerate(
                    tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0, desc='Testing classification')):
                categorical_fields, numerical_fields, labels = categorical_fields.to(device), numerical_fields.to(
                    device), labels.to(device)
                y = model(categorical_fields, numerical_fields)
                for i in range(task_num):
                    labels_dict[i].extend(labels[:, i].tolist())
                    predicts_dict[i].extend(y[i].tolist())
                    loss_dict[i].extend(
                        torch.nn.functional.binary_cross_entropy(y[i], labels[:, i].float(), reduction='none').tolist())
                stop_iteration = False
                for callback in step_callbacks or []:
                    try:
                        callback(step=j)
                    except StopIteration:
                        stop_iteration = True
                if stop_iteration:
                    break
        sco_results, loss_results = list(), list()
        for i in range(task_num):
            sco_results.append(roc_auc_score(labels_dict[i], predicts_dict[i]))
            loss_results.append(np.array(loss_dict[i]))
            tensorboard.add_pr_curve(f'task{task_num}',
                                     labels=labels_dict[i],
                                     predictions=predicts_dict[i],
                                     global_step=epoch)
        return sco_results, loss_results

    elif loss_type == 'MSELoss'.lower():
        model.eval()
        labels_dict, predicts_dict, loss_dict = {}, {}, {}
        for i in range(task_num):
            labels_dict[i], predicts_dict[i], loss_dict[i] = list(), list(), list()
        with torch.no_grad():
            for j, (categorical_fields, numerical_fields, labels) in enumerate(
                    tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0, desc='Testing regression')):
                categorical_fields, numerical_fields, labels = categorical_fields.to(device), numerical_fields.to(
                    device), labels.to(device)
                y = model(categorical_fields, numerical_fields)
                for i in range(task_num):
                    labels_dict[i].extend(labels[:, i].tolist())
                    predicts_dict[i].extend(y[i].tolist())
                    loss_dict[i].append(
                        float(torch.nn.functional.mse_loss(y[i], labels[:, i].float()).detach().cpu().numpy()))
                stop_iteration = False
                for callback in step_callbacks or []:
                    try:
                        callback(step=j)
                    except StopIteration:
                        stop_iteration = True
                if stop_iteration:
                    break
        sco_results, loss_results = list(), list()
        for i in range(task_num):
            sco_results.append(r2_score(labels_dict[i], predicts_dict[i]))
            loss_results.append(np.array(loss_dict[i]))
        return sco_results, loss_results

# %%
