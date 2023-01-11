from abc import abstractmethod
import numpy as np
import tqdm
import wandb
from tensorboardX import SummaryWriter
tensorboard = SummaryWriter('./tensorboard_log')

def get_trainer(model_name, do_balance, **kargs):
    if model_name == 'metaheac':
        assert not do_balance
        return MetaTrainer(**kargs)
    else:
        return DefaultTrainer(**kargs) if not do_balance else BalanceTrainer(**kargs)
        
class MultitaskTrainer:
    @abstractmethod
    def train_epoch(self):
        pass

class DefaultTrainer(MultitaskTrainer):
    def __init__(self, model, optimizer, data_loader, criterion, device, log_interval=100, step_callbacks=None, **kargs) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.criterion = criterion
        self.device = device
        self.log_interval = log_interval
        self.epoch = 0
        self.step_callbacks = step_callbacks
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        epoch_losses = 0
        loader = tqdm.tqdm(self.data_loader, smoothing=0, mininterval=1.0)
        for i, (categorical_fields, numerical_fields, labels) in enumerate(loader):
            categorical_fields, numerical_fields, labels = categorical_fields.to(self.device), numerical_fields.to(self.device), labels.to(self.device)
            y = self.model(categorical_fields, numerical_fields)
            loss_list = [self.criterion(y[i], labels[:, i].float()) for i in range(labels.size(1))]
            loss = 0
            for item in loss_list:
                loss += item
            loss /= len(loss_list)
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            if (i + 1) % self.log_interval == 0:
                loader.set_postfix(loss=total_loss / self.log_interval)
                wandb.log({'step_loss':total_loss / self.log_interval, 'epoch': self.epoch, 'step': i})
                tensorboard.add_scalar('step_loss', total_loss / self.log_interval, self.epoch*len(loader)+i)
                epoch_losses = loss_list
                total_loss = 0
            stop_iteration = False
            for callback in self.step_callbacks or []:
                try:
                    callback(step=i)
                except StopIteration:
                    stop_iteration = True
            if stop_iteration:
                break
        self.epoch +=1
        return epoch_losses


class MetaTrainer(MultitaskTrainer):
    def __init__(self, model, optimizer, data_loader, device, log_interval=100, step_callbacks=None, **kargs) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.device = device
        self.log_interval = log_interval
        self.epoch = 0
        self.step_callbacks = step_callbacks
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        epoch_losses = 0
        loader = tqdm.tqdm(self.data_loader, smoothing=0, mininterval=1.0)
        list_sup_categorical, list_sup_numerical, list_sup_y, list_qry_categorical, list_qry_numerical, list_qry_y = list(), list(), list(), list(), list(), list()
        for i, (categorical_fields, numerical_fields, labels) in enumerate(loader):
            categorical_fields, numerical_fields, labels = categorical_fields.to(self.device), numerical_fields.to(self.device), labels.to(self.device)
            batch_size = int(categorical_fields.size(0) / 2)
            list_sup_categorical.append(categorical_fields[:batch_size])
            list_qry_categorical.append(categorical_fields[batch_size:])
            list_sup_numerical.append(numerical_fields[:batch_size])
            list_qry_numerical.append(numerical_fields[batch_size:])
            list_sup_y.append(labels[:batch_size])
            list_qry_y.append(labels[batch_size:])
            
            if (i + 1) % 2 == 0:
                loss = self.model.global_update(list_sup_categorical, list_sup_numerical, list_sup_y, list_qry_categorical, list_qry_numerical, list_qry_y)
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                list_sup_categorical, list_sup_numerical, list_sup_y, list_qry_categorical, list_qry_numerical, list_qry_y = list(), list(), list(), list(), list(), list()
            if (i + 1) % self.log_interval == 0:
                loader.set_postfix(loss=total_loss / self.log_interval)
                wandb.log({'step_loss':total_loss / self.log_interval, 'epoch': self.epoch, 'step': i})
                tensorboard.add_scalar('step_loss', total_loss / self.log_interval, self.epoch*len(loader)+i)
                epoch_losses  = total_loss
                total_loss = 0
            stop_iteration = False
            for callback in self.step_callbacks or []:
                try:
                    callback(step=i)
                except StopIteration:
                    stop_iteration = True
            if stop_iteration:
                break
        self.epoch +=1
        return [epoch_losses]

    

class BalanceTrainer(MultitaskTrainer):
    def __init__(self, model, optimizer_taskLayer, optimizer_sharedLayer, multitask_balancer, data_loader, 
                 criterion, device, log_interval=100, step_callbacks=None, **kargs) -> None:
        super().__init__()
        self.model = model
        self.optimizer_taskLayer = optimizer_taskLayer
        self.optimizer_sharedLayer = optimizer_sharedLayer
        self.multitask_balancer = multitask_balancer
        self.data_loader = data_loader
        self.criterion = criterion
        self.device = device
        self.log_interval = log_interval
        self.epoch = 0
        self.step_callbacks = step_callbacks
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_losses = np.array([0, 0])
        tqdmloader = tqdm.tqdm(self.data_loader, smoothing=0, mininterval=1.0)
        for i, (categorical_fields, numerical_fields, labels) in enumerate(tqdmloader):
            # 从数据集中读取数据，计算输出和loss。
            categorical_fields, numerical_fields, labels = categorical_fields.to(
                self.device), numerical_fields.to(self.device), labels.to(self.device)
            y = self.model(categorical_fields, numerical_fields)
            loss_list = [self.criterion(y[i], labels[:, i].float())
                        for i in range(labels.size(1))]
            loss = 0
            for j, item in enumerate(loss_list):
                loss += item
                total_losses[j]+= (item.detach().cpu().numpy()- total_losses[j])/(i+1)
            loss /= len(loss_list)

            # 似乎必须这样写
            # 这是满足 规律的吗？  可能变大了一倍，但是不影响结果。
            self.model.zero_grad()  # 清空上一轮训练的梯度。
            loss.backward(retain_graph=True)
            self.multitask_balancer.step(loss_list)        
            self.optimizer_taskLayer.step()
            self.optimizer_sharedLayer.step()

            total_loss += loss.item()
            if (i + 1) % self.log_interval == 0:
                tqdmloader.set_postfix(loss=total_loss / self.log_interval)
                wandb.log({'step_loss':total_loss / self.log_interval, 'epoch': self.epoch, 'step': i})
                tensorboard.add_scalar('step_loss', total_loss / self.log_interval, self.epoch*len(tqdmloader)+i)
                total_loss = 0
            stop_iteration = False
            for callback in self.step_callbacks or []:
                try:
                    callback(step=i)
                except StopIteration:
                    stop_iteration = True
            if stop_iteration:
                break
        self.epoch +=1
        return total_losses

    