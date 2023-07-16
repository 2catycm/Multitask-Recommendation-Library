# To run train.py, use this code: python train.py --params 实验/NL_basic.yaml
# %%

import re
import global_vars
from tensorboardX import SummaryWriter
global_vars.tensorboard = SummaryWriter('./tensorboard_log')

import argparse
import wandb
from libauc.sampler import DualSampler
from models.abstract_multitask_model import MultitaskWrapper
from utils.gpu_manager import GPUManager
from pathlib import Path
import torch
import tqdm
from torch.utils.data import DataLoader
import os
import numpy as np
from callbacks.debug_step_callback import JustTestCanRun
# from imblearn.over_sampling import RandomOverSampler
from sampler.sampler import ClassAwareSampler


from datasets import get_dataset
from models import get_model
from losses import get_loss
from trainers import get_trainer
from multi_balancer import get_multi_balancer
from optimizers import get_optimizer, get_optimizer0


from utils.early_stopper import EarlyStopper
from utils import torch_utils
from utils.general import LOGGER, colorstr, yaml_load, check_git_info, init_seeds

from val import *

from munch import DefaultMunch, Munch


# from accelerate import Accelerator
# accelerator = Accelerator()
from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# %%


def set_seeds(params):
    if params.get('deterministic', False):
        LOGGER.info(f"实验模式为{colorstr('确定性实验')}，将会设置固定随机种子。")
        torch_utils.make_exp_reproducible(params.seed or 3407)
    else:
        LOGGER.info(f"实验模式为{colorstr('统计性实验')}，随机种子不做设置。")
    # https://pytorch.org/docs/stable/elastic/run.html
    # LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
    # RANK = int(os.getenv('RANK', -1))
    # WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

# %%


def select_device(device_num):
    if device_num == 'auto':
        LOGGER.info(f"训练设备：自动模式寻找中...")
        gm = GPUManager()
        return torch.device(gm.auto_choice())
    elif device_num.startswith('cuda'):
        return torch.device(device_num)
    else:
        try:
            device_num = int(device_num)
            return torch.device(device_num)
        except:
            try:
                return select_device('auto'), list(map(int, re.split(',\s*', device_num)))
            except Exception as e:
                raise ValueError(f"无法识别的设备编号{device_num}。\n{e}")

# %%


def main(params: Munch):
    set_seeds(params)
    device = select_device(params.device_num)
    if isinstance(device, tuple):
        device, device_ids = device
        # 多卡情况
        LOGGER.info(f"{colorstr('训练设备')}: {params.device_num}。")
    else:
        LOGGER.info(f"{colorstr('训练设备')}: {device}。")
    # 1. 数据集
    LOGGER.info(f"{colorstr('数据集')}: 开始加载{params.dataset_name}: ")
    train_dataset = get_dataset(params.dataset_type, params.train_path)
    test_dataset = get_dataset(params.dataset_type, params.test_path)

    if params.get('sampling', False):
        
        sampler = ClassAwareSampler(train_dataset, num_samples_cls=4) # TODO 使用sampler进行采样
        train_data_loader = DataLoader(train_dataset, batch_size=params.batch_size,  sampler=sampler,
                                    num_workers=16, shuffle=False)
                                    #    num_workers=16, shuffle=True, pin_memory=True)
    else:
        train_data_loader = DataLoader(train_dataset, batch_size=params.batch_size, 
                                    num_workers=16, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=params.batch_size,
                                  num_workers=16, shuffle=False)  # num_workers是GPU数量的四倍。
                                #   num_workers=16, shuffle=False, pin_memory=True)  # num_workers是GPU数量的四倍。

    LOGGER.info(f"{colorstr('数据集')}: 加载成功。")
    # 2. 根据数据集信息以及模型选择获取模型
    LOGGER.info(f"{colorstr('模型')}: 开始加载{params.model_name}: ")
    field_dims = train_dataset.field_dims
    numerical_num = train_dataset.numerical_num
    task_num = train_dataset.labels_num
    model = get_model(params.model_name, field_dims, numerical_num, task_num,
                      params.expert_num, params.embed_dim,
                      bottom_mlp_dims=params.bottom_mlp_dims, tower_mlp_dims=params.tower_mlp_dims,
                      dropout=params.dropout,
                      )

    # if 'compile' in dir(torch):
    #     LOGGER.info(f"{colorstr('Pytorch 2.0')} detected, compiling the model to speed up. ")
    #     # model = torch.compile(model, mode="max-autotune")
    #     # model = torch.compile(model, mode="max-autotune", fullgraph=True, backend='Eager')
    #     model = torch.compile(model, mode="max-autotune", fullgraph=True)
    #     LOGGER.info(f"{colorstr('Pytorch 2.0')} compilation done. ")
    # dummy_input = (torch.zeros((params.batch_size, train_dataset.categorical_num)).to(torch.int64),
    #                torch.zeros((params.batch_size, train_dataset.numerical_num)).to(torch.float32))
    wrapped = MultitaskWrapper(model).to('cpu')
    # dummy_input = (torch.zeros((2, train_dataset.categorical_num)).to(torch.int64),
    #                torch.zeros((2, train_dataset.numerical_num)).to(torch.float32))
    size = (train_dataset.categorical_num+train_dataset.numerical_num, )
    dummy_input = torch.zeros((2, *size)).to('cpu')
    tensorboard.add_graph(wrapped, input_to_model=dummy_input)
    # print(model)
    wrapped = wrapped.to('cuda')
    print(summary(wrapped, size))
    LOGGER.info(f"{colorstr('模型')}: 加载成功。")

    # if not isinstance(device, list):
    model = model.to(device)

    # weights 迁移 if
    weights = Path(params.weights or '').resolve()
    if weights.is_file() and weights.exists():
        state_dict = torch.load(weights)
        model.load_state_dict(state_dict)
        LOGGER.info(f"{colorstr('模型')}: 迁移参数完成。")
    # TODO DP或者DDP

    # 3. 损失函数： TODO 根据多任务获得离散型和连续型的损失函数
    criterion = get_loss(params.categorical_loss, device=device)
    # criterion = criterion.to(device)
    LOGGER.info(f"{colorstr('损失函数')}: 加载完成：{criterion}。 ")
    # LOGGER.info(f"{colorstr('损失函数')}: 加载完成。 ")

    # 4. callback
    early_stopper = EarlyStopper(
        params.patience, params.min_delta, params.cumulative_delta)
    step_callbacks = []
    if params.get("just_test_can_run", False):
        LOGGER.info(f"{colorstr('软件测试')}: 当前为软件测试模式， 只是验证代码是否能够运行。 ")
        step_callbacks.append(JustTestCanRun())

    # 5. 优化器
    if params.do_balance:
        optimizer_sharedLayer = get_optimizer0(
            model_weights=model.shared_parameters(), **params)
        optimizer_taskLayer = get_optimizer0(
            model_weights=model.specific_parameters(), **params)
        # 多任务优化器
        multitask_balancer = get_multi_balancer(params.balancer_name,
                                                model.shared_parameters(),
                                                params.corr_factor)
        LOGGER.info(f"{colorstr('优化器')}: 加载完成，当前为平衡模式。 ")
        # 5. 获得训练器
        if 'device_ids' in locals().keys():
            model = nn.DataParallel(model, device_ids=device_ids)
        
        trainer = get_trainer(params.model_name, model=model,
                              optimizer_sharedLayer=optimizer_sharedLayer,
                              optimizer_taskLayer=optimizer_taskLayer,
                              multitask_balancer=multitask_balancer,
                              data_loader=train_data_loader, criterion=criterion,
                              device=device, do_balance=params.do_balance,
                              step_callbacks=step_callbacks)
    else:
        optimizer = get_optimizer(model_weights=model.parameters(), model=model,
                                  criti=criterion, device=device, **params)
        # LOGGER.info(f"{colorstr('优化器')}: 加载完成：{optimizer}。 ")
        LOGGER.info(f"{colorstr('优化器')}: 加载完成。 ")
        # 6. 获得训练器
        if isinstance(device, list):
            model = nn.DataParallel(model, device_ids=device)
        trainer = get_trainer(params.model_name, model=model,
                              optimizer=optimizer, data_loader=train_data_loader,
                              criterion=criterion, device=device, do_balance=params.do_balance,
                              step_callbacks=step_callbacks)
        # model, optimizer, train_data_loader = accelerator.prepare(model, optimizer, train_data_loader)

    # 创建新的实验记录
    save_dir = Path(params.save_dir).resolve().absolute()
    i = 0
    while True:
        exp_save_dir = save_dir/f"exp_{i}"
        if not exp_save_dir.exists():
            exp_save_dir.mkdir(parents=True)
            save_dir = exp_save_dir
            LOGGER.info(
                f"{colorstr('实验管理')}: 数据集{params.dataset_name}新增实验{i}。")
            if params.clearml:
                from clearml import Task
                clearml_task = Task.init(project_name='test_multitask' if params.just_test_can_run else params.dataset_name,
                                         task_name=f'{params.model_name}+{params.categorical_loss}+{i}')
                # clearml_task.get_logger().report_
                clearml_task.connect_configuration(configuration=dict(params))
            break
        i += 1

    LOGGER.info(f"{colorstr('训练')}: 终于可以开始啦! ")
    
    
    for epoch_i in range(params.max_epochs):
        epoch_losses = trainer.train_epoch()
        if params.categorical_loss.lower() == 'AUCMLoss'.lower():
            optimizer.update_regularizer()
        # epoch_losses = list(map(lambda x:x.detach().cpu().numpy(), epoch_losses)) # requires grad要detach

        # 在这里作了是否使用BCELoss的讨论，从而不使用auc，代码正确性未能保证 //from oyl
        scores, losses = test(model, test_data_loader, task_num,
                              device, epoch=epoch_i, step_callbacks=step_callbacks,
                              loss_type=params.categorical_loss.lower())

        sco_data = {'avg_sco': np.array(scores).mean(), 'epoch': epoch_i}
        loss_data = {'avg_loss': np.array(losses).mean(), 'epoch': epoch_i}
        tensorboard.add_scalar(f'avg_sco', np.array(scores).mean(), epoch_i)
        tensorboard.add_scalar(
            f'avg_val_loss', np.array(losses).mean(), epoch_i)
        # train_loss_data = {'train_loss':np.array(epoch_losses).mean(), 'epoch': epoch_i}
        for i in range(task_num):
            # LOGGER.info(f'task {i}, Score {scores[i]}, Log-loss {losses[i]}')
            sco_data[f'score{i}'] = scores[i]
            loss_data[f'loss{i}'] = losses[i][-1]
            tensorboard.add_scalar(f'score{i}', scores[i], epoch_i)
            tensorboard.add_scalar(f'val_loss{i}', losses[i][-1], epoch_i)
            # train_loss_data[f'loss{i}'] = epoch_losses[i] TODO 多任务loss
        if params.wandb:
            wandb.log(sco_data)
            wandb.log(loss_data)
            # wandb.log(train_loss_data)

        if epoch_i % params.save_epoch == 0:
            save_path = save_dir / f"{params.model_name}_{epoch_i}.pt"
            torch.save(model.state_dict(), save_path)
            LOGGER.info(f"latest model saved to {save_path}")

        if not early_stopper.is_continuable(epoch_i, np.array(scores).mean()):
            LOGGER.info(
                f"Early Stopper在{early_stopper.patience_counter}轮都没有改进后丧失了耐心，训练即将结束。")
            break

    LOGGER.info(
        f'历史最佳 {early_stopper.best_score} 在第 {early_stopper.best_epoch} 轮达到。')


def read_python_config(path, default_path="实验/default.py"):
    exec(open(default_path).read())
    # default_config = {k:v for k,v in locals().items() if not k.startswith('_')}
    exec(open(path).read())
    new_config = {k: v for k, v in locals().items() if not k.startswith('_')}
    return DefaultMunch.fromDict(new_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--param', '--params', '-p',
                        # default='实验/NL_basic.yaml')
                        # default='实验/default.py')
                        # default='实验/default.yaml')
                        # default='实验/ple_run_shallow.yaml')
                        default='实验/dense_ple.yaml')
                        # default='实验/dense_ple_test.yaml')
    # parser.add_argument('--param', '--params', '-p', default='实验/test/metaheac_loss_new.yaml')
    args = parser.parse_args()
    args = yaml_load(args.param)
    # override default parameters with custom yaml file
    default = yaml_load('实验/default.yaml')
    default.update(args)
    params = DefaultMunch.fromDict(default)
    # params = read_python_config(args.param)

    def print_params(params):
        LOGGER.info(colorstr('parameters: ') +
                    ', '.join(f'{k}={v}' for k, v in params.items()))
    global_vars.wandb = params.get("wandb", False)
    global_vars.clearml = params.get("clearml", False)
    
    if global_vars.wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project=params.project,
            name=params.experiment_name
        )
        wandb.config.update(params)  # 可能会做一些调整
        print_params(wandb.config)
        main(wandb.config)
    else:
        print_params(params)
        main(params)


# %%
