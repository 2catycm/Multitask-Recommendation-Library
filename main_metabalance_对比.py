# To run main_metabalance.py, use this code:
# python main_metabalance.py --model_name mmoe --expert_num 8 --dataset_name Synthetic --dataset_path ./synthetic_datasets

from torch.utils.data import random_split
from tensorboardX import SummaryWriter
import torch
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import os
import numpy as np

from datasets.aliexpress import AliExpressDataset
from datasets.syndataset import SynDataset
from models.sharedbottom import SharedBottomModel
from models.singletask import SingleTaskModel
from models.omoe import OMoEModel
from models.mmoe import MMoEModel
from models.ple import PLEModel
from models.aitm import AITMModel
from models.metaheac import MetaHeacModel

from optimizer.metabalance import MetaBalance
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)

def get_dataset(name, path):
    if 'AliExpress' in name:
        return AliExpressDataset(path)
    elif 'Synthetic' in name:
        return SynDataset(path)
    else:
        raise ValueError('unknown dataset name: ' + name)


def get_model(name, categorical_field_dims, numerical_num, task_num, expert_num, embed_dim):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    if name == 'sharedbottom':
        print("Model: Shared-Bottom")
        return SharedBottomModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64), task_num=task_num, dropout=0.2)
    elif name == 'singletask':
        print("Model: SingleTask")
        return SingleTaskModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64), task_num=task_num, dropout=0.2)
    elif name == 'omoe':
        print("Model: OMoE")
        return OMoEModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64), task_num=task_num, expert_num=expert_num, dropout=0.2)
    elif name == 'mmoe':
        print("Model: MMoE")
        return MMoEModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64), task_num=task_num, expert_num=expert_num, dropout=0.2)
    elif name == 'ple':
        print("Model: PLE")
        return PLEModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64), task_num=task_num, shared_expert_num=int(expert_num / 2), specific_expert_num=int(expert_num / 2), dropout=0.2)
    elif name == 'aitm':
        print("Model: AITM")
        return AITMModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64), task_num=task_num, dropout=0.2)
    elif name == 'metaheac':
        print("Model: MetaHeac")
        return MetaHeacModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64), task_num=task_num, expert_num=expert_num, critic_num=5, dropout=0.2)
    else:
        raise ValueError('unknown model name: ' + name)


class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model.state_dict(), self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


def metatrain(model, optimizer, data_loader, device, log_interval=100):
    model.train()
    total_loss = 0
    loader = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    list_sup_categorical, list_sup_numerical, list_sup_y, list_qry_categorical, list_qry_numerical, list_qry_y = list(
    ), list(), list(), list(), list(), list()
    for i, (categorical_fields, numerical_fields, labels) in enumerate(loader):
        categorical_fields, numerical_fields, labels = categorical_fields.to(
            device), numerical_fields.to(device), labels.to(device)
        batch_size = int(categorical_fields.size(0) / 2)
        list_sup_categorical.append(categorical_fields[:batch_size])
        list_qry_categorical.append(categorical_fields[batch_size:])
        list_sup_numerical.append(numerical_fields[:batch_size])
        list_qry_numerical.append(numerical_fields[batch_size:])
        list_sup_y.append(labels[:batch_size])
        list_qry_y.append(labels[batch_size:])

        if (i + 1) % 2 == 0:
            loss = model.global_update(list_sup_categorical, list_sup_numerical,
                                       list_sup_y, list_qry_categorical, list_qry_numerical, list_qry_y)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            list_sup_categorical, list_sup_numerical, list_sup_y, list_qry_categorical, list_qry_numerical, list_qry_y = list(
            ), list(), list(), list(), list(), list()
        if (i + 1) % log_interval == 0:
            loader.set_postfix(loss=total_loss / log_interval)
            total_loss = 0
tensorboard = SummaryWriter('./tensorboard_log')


def test(model, data_loader, task_num, criterion, device):
    model.eval()
    loss_lists = torch.Tensor([0, 0]).to(device)
    loader = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (categorical_fields, numerical_fields, labels) in enumerate(loader):
        categorical_fields, numerical_fields, labels = categorical_fields.to(
            device), numerical_fields.to(device), labels.to(device)
        y = model(categorical_fields, numerical_fields)
        loss_list = [criterion(y[i], labels[:, i].float())
                     for i in range(labels.size(1))]
        loss = 0
        for i, item in enumerate(loss_list):
            loss_lists[i] += item
            
    return loss_lists/len(data_loader)


def train(model, optimizer, data_loader, criterion, device, log_interval=1):
    model.train()
    total_loss = 0
    total_losses = np.array([0, 0])
    loader = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (categorical_fields, numerical_fields, labels) in enumerate(loader):
        categorical_fields, numerical_fields, labels = categorical_fields.to(
            device), numerical_fields.to(device), labels.to(device)
        y = model(categorical_fields, numerical_fields)
        loss_list = [criterion(y[i], labels[:, i].float())
                     for i in range(labels.size(1))]
        loss = 0
        for j, item in enumerate(loss_list):
            loss += item
            total_losses[j]+= (item.detach().cpu().numpy()- total_losses[j])/(i+1)
            
        loss /= len(loss_list)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            loader.set_postfix(loss=total_loss / log_interval)
            total_loss = 0
    return total_losses



def train_metabalance(model, optimizer_taskLayer, optimizer_sharedLayer, metabalance,
                      data_loader, criterion, device, log_interval=1):
    model.train()
    total_loss = 0
    total_losses = np.array([0, 0])
    tqdmloader = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (categorical_fields, numerical_fields, labels) in enumerate(tqdmloader):
        # 从数据集中读取数据，计算输出和loss。
        categorical_fields, numerical_fields, labels = categorical_fields.to(
            device), numerical_fields.to(device), labels.to(device)
        y = model(categorical_fields, numerical_fields)
        loss_list = [criterion(y[i], labels[:, i].float())
                     for i in range(labels.size(1))]
        loss = 0
        for j, item in enumerate(loss_list):
            loss += item
            total_losses[j]+= (item.detach().cpu().numpy()- total_losses[j])/(i+1)
        loss /= len(loss_list)

        # # 更新task layer。
        # model.zero_grad()  # 清空上一轮训练的梯度。
        # loss.backward(retain_graph=True)
        # optimizer_taskLayer.step()

        # # 更新shared layer。
        # model.zero_grad()  # 我们重新根据4个loss计算梯度，不用上面那个梯度。
        # metabalance.step(loss_list)
        # optimizer_sharedLayer.step()
        
        
        # 似乎必须这样写
        # 这是满足 规律的吗？  可能变大了一倍，但是不影响结果。
        model.zero_grad()  # 清空上一轮训练的梯度。
        loss.backward(retain_graph=True)
        metabalance.step(loss_list)        
        optimizer_taskLayer.step()
        optimizer_sharedLayer.step()

        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tqdmloader.set_postfix(loss=total_loss / log_interval)
            total_loss = 0
    return total_losses


def main(dataset_name,
         dataset_path,
         task_num,
         expert_num,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         embed_dim,
         weight_decay,
         device,
         save_dir):
    device = torch.device(device)
    dataset = get_dataset(dataset_name, dataset_path + '/syn_data1.csv')
    print(len(dataset))
    # 把一个dataset变成train和test
    train_dataset, test_dataset = random_split(
        dataset=dataset,
        lengths=[0.8, 0.2],
        generator=torch.Generator().manual_seed(0)
    )
    print(len(train_dataset))

    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    test_data_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    print("Finished loading data")
    field_dims = dataset.field_dims
    numerical_num = dataset.numerical_num

    # 获取模型。
    model = get_model(model_name, field_dims, numerical_num,
                      task_num, expert_num, embed_dim).to(device)

    # 获得loss
    # criterion = torch.nn.BCELoss()
    criterion = torch.nn.MSELoss()

    # 获得优化器。
    metabalance = MetaBalance(model.shared_parameters())
    optimizer_sharedLayer = torch.optim.Adam(
        model.shared_parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_taskLayer = torch.optim.Adam(
        model.specific_parameters(), lr=learning_rate, weight_decay=weight_decay)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay)

    # 存储模型。
    save_path = f'{save_dir}/{dataset_name}_{model_name}.pt'
    early_stopper = EarlyStopper(num_trials=1000, save_path=save_path)
    print("begin to train")

    f0 = open('{}_{}0.txt'.format(
        model_name, dataset_name), 'a', encoding='utf-8')

    for epoch_i in range(epoch):
        if model_name == 'metaheac':
            metatrain(model, optimizer, train_data_loader, device)
            
        else:
            # train_lossed = train_metabalance(model, optimizer_taskLayer, optimizer_sharedLayer,
            #                   metabalance, train_data_loader, criterion, device)

            
            train_lossed = train(model, optimizer,
                                 train_data_loader, criterion, device)
            
            
            tensorboard.add_scalar('train-loss-total', train_lossed.mean(), epoch_i)
            for j, loss in enumerate(train_lossed):
                tensorboard.add_scalar('train-loss' + str(j), loss, epoch_i)

        losses = test(model, test_data_loader, task_num, criterion, device)
        
        for j, loss in enumerate(losses):
            tensorboard.add_scalar('test-loss' + str(j), loss, epoch_i)
        tensorboard.add_scalar('test-loss-total', losses.mean(), epoch_i)
        
        print('epoch:', epoch_i, 'test: loss:', losses.mean())

        f0.write('epoch {}, loss: {}\n'.format(epoch_i, losses.mean()))

        for i in range(task_num):
            print('task {}, loss {}'.format(i, losses[i]))
        if not early_stopper.is_continuable(model, losses.mean()):
            print(f'test: best loss: {early_stopper.best_accuracy}')
            break

    f0.write('\n')
    f0.close()

    model.load_state_dict(torch.load(save_path))
    # auc, loss = test(model, test_data_loader, task_num, device)
    f = open('{}_{}.txt'.format(model_name, dataset_name),
             'a', encoding='utf-8')
    f.write('learning rate: {}\n'.format(learning_rate))
    for i in range(task_num):
        print('task {}, Log-loss {}'.format(i, losses[i]))
        f.write('task {}, Log-loss {}\n'.format(i, losses[i]))
    print('\n')
    f.write('\n')
    f.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset_name', default='AliExpress_NL', choices=['AliExpress_NL', 'AliExpress_ES', 'AliExpress_FR', 'AliExpress_US', 'Synthetic'])
    parser.add_argument('--dataset_name', default='Synthetic', choices=[
                        'AliExpress_NL', 'AliExpress_ES', 'AliExpress_FR', 'AliExpress_US', 'Synthetic'])
    # parser.add_argument('--dataset_path', default='./data/')
    parser.add_argument('--dataset_path', default='./synthetic_datasets')
    parser.add_argument('--model_name', default='mmoe', choices=[
                        'singletask', 'sharedbottom', 'omoe', 'mmoe', 'ple', 'aitm', 'metaheac'])
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--task_num', type=int, default=2)
    parser.add_argument('--expert_num', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--embed_dim', type=int, default=128)
    # parser.add_argument('--embed_dim', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='chkpt')
    args = parser.parse_args()
    print("begin main")
    main(args.dataset_name,
         args.dataset_path,
         args.task_num,
         args.expert_num,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.embed_dim,
         args.weight_decay,
         args.device,
         args.save_dir)
