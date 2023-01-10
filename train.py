# To run main.py, use this code: python train.py --params 实验/NL_basic.yaml
#%%
from pathlib import Path
import torch
import tqdm
from torch.utils.data import DataLoader
import os
import numpy as np

from datasets import get_dataset
from models import get_model
from losses import get_loss
from trainers import get_trainer
from multi_balancer import get_multi_balancer
from optimizers import get_optimizer


from utils.early_stopper import EarlyStopper
from utils import torch_utils 
from utils.general import LOGGER, colorstr, yaml_load, check_git_info, init_seeds

from val import *

from munch import DefaultMunch, Munch
#%%
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
# torch_utils.make_exp_reproducible()
init_seeds(3407 + 1 + RANK, deterministic=True)


#%%
def main(params:Munch):
    # device = torch_utils.select_device(device)
    device = torch.device('cuda')
    # 1. 数据集
    LOGGER.info(f"{colorstr('数据集')}: 开始加载{params.dataset_name}: ")
    train_dataset = get_dataset(params.dataset_name, os.path.join(params.dataset_path, params.dataset_name) + '/train.csv')
    test_dataset = get_dataset(params.dataset_name, os.path.join(params.dataset_path, params.dataset_name) + '/test.csv')
    train_data_loader = DataLoader(train_dataset, batch_size=params.batch_size, num_workers=4, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=params.batch_size, num_workers=4, shuffle=False)
    LOGGER.info(f"{colorstr('数据集')}: 加载成功。")
    # 2. 根据数据集信息以及模型选择获取模型
    LOGGER.info(f"{colorstr('模型')}: 开始加载{params.model_name}: ")
    field_dims = train_dataset.field_dims
    numerical_num = train_dataset.numerical_num
    task_num = train_dataset.labels_num
    model = get_model(params.model_name, field_dims, numerical_num, task_num, 
                      params.expert_num, params.embed_dim).to(device)
    if 'compile' in dir(torch):
        LOGGER.info(f"{colorstr('Pytorch 2.0')} detected, compiling the model to speed up. ")
        model = torch.compile(model, mode="max-autotune")
        # model = torch.compile(model, mode="max-autotune", fullgraph=True, backend='Eager')
        # model = torch.compile(model, mode="max-autotune", fullgraph=True)
        LOGGER.info(f"{colorstr('Pytorch 2.0')} compilation done. ")
        
    # TODO weights 迁移 if 
    # TODO DP或者DDP
    LOGGER.info(f"{colorstr('模型')}: 加载成功。")
    
    # 3. 损失函数： TODO 根据多任务获得离散型和连续型的损失函数
    criterion = get_loss(params.categorical_loss)
    LOGGER.info(f"{colorstr('损失函数')}: 加载完成：{criterion}。 ")
        
    # 4. 优化器
    if params.do_balance:
        # optimizer_sharedLayer = get_optimizer(optimizer_name=params.optimizer_name, model.shared_parameters(), **params)
        # optimizer_taskLayer = get_optimizer(optimizer_name=params.optimizer_name, model.specific_parameters(), **params)
        # # 多任务优化器
        # multitask_optimizer = get_multi_balancer(balancer_name=params.balancer_name, model.shared_parameters(), **params)
        # LOGGER.info(f"{colorstr('优化器')}: 加载完成，当前为平衡模式。 ")    
        # # 5. 获得训练器
        # trainer = get_trainer(params.model_name, model=model, 
        #                     optimizer_sharedLayer=optimizer_sharedLayer,
        #                     optimizer_taskLayer = optimizer_taskLayer,
        #                     multitask_optimizer=multitask_optimizer, 
        #                     data_loader=train_data_loader, criterion=criterion, 
        #                     device=device, do_balance=params.do_balance)    
        pass
    else:
        optimizer = get_optimizer(model_weights=model.parameters(), **params)
        LOGGER.info(f"{colorstr('优化器')}: 加载完成：{optimizer}。 ")
        # 5. 获得训练器
        trainer = get_trainer(params.model_name, model=model, 
                            optimizer=optimizer, data_loader=train_data_loader, 
                            criterion=criterion, device=device, do_balance=params.do_balance)

    early_stopper = EarlyStopper(params.patience, params.min_delta, params.cumulative_delta)
    
    # 创建新的实验记录
    save_dir = Path(params.save_dir).resolve().absolute()
    i= 0
    while True:
        exp_save_dir = save_dir/f"exp_{i}"
        if not exp_save_dir.exists():
            exp_save_dir.mkdir(parents=True)
            save_dir = exp_save_dir
            LOGGER.info(f"{colorstr('实验管理')}: 数据集{params.dataset_name}新增实验{i}。")    
            break
        i+=1
    
    LOGGER.info(f"{colorstr('训练')}: 终于可以开始啦! ")    
    for epoch_i in range(params.max_epochs):
        trainer.train_epoch()
        auc, loss = test(model, test_data_loader, task_num, device)
        LOGGER.info('epoch:', epoch_i, 'test: auc:', auc)

        if epoch_i%params.save_epoch==0:
            save_path = save_dir / f"{params.model_name}_{epoch_i}.pt"
            torch.save(model.state_dict(), save_path)
            LOGGER.info(f"latest model saved to {save_path}")
        for i in range(task_num):
            LOGGER.info(f'task {i}, AUC {auc[i]}, Log-loss {loss[i]}')
            
        if not early_stopper.is_continuable(epoch_i, np.array(auc).mean()):
            LOGGER.info(f"Early Stopper在{early_stopper.patience_counter}轮都没有改进后丧失了耐心，训练即将结束。")
            break

    LOGGER.info(f'历史最佳AUC {early_stopper.best_score} 在第 {early_stopper.best_epoch} 轮达到。')
    
    

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--param', '--params', '-p', default='实验/NL_basic.yaml')
    args = parser.parse_args()
    args = yaml_load(args.param)
    params = DefaultMunch.fromDict(args)    
    LOGGER.info(colorstr('parameters: ') + ', '.join(f'{k}={v}' for k, v in params.items()))
    main(params)
    
# %%
