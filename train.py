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
from trainers import get_trainer
from multitask_optimizers import get_multitask_optimizers


from libauc.losses import AUCMLoss
from libauc.optimizers import PESG

from utils.early_stopper import EarlyStopper
from utils import torch_utils 
from utils.general import LOGGER, colorstr, yaml_load

from val import *

from munch import DefaultMunch
#%%
def main(dataset_name,dataset_path,task_num,
         expert_num,model_name,epoch,
         learning_rate,batch_size,embed_dim,
         weight_decay,device,save_dir, 
         **kwargs):
    """_summary_

    Args:
        dataset_name (_type_): _description_
        dataset_path (_type_): _description_
        task_num (_type_): _description_
        expert_num (_type_): _description_
        model_name (_type_): _description_
        epoch (_type_): _description_
        learning_rate (_type_): _description_
        batch_size (_type_): _description_
        embed_dim (_type_): _description_
        weight_decay (_type_): _description_
        device (_type_): _description_
        save_dir (_type_): _description_
    """
    kwargs = DefaultMunch.fromDict(kwargs)    
    torch_utils.make_exp_reproducible()
    device = torch_utils.select_device(device)
    # 1. 数据集
    LOGGER.info("Start loading data")
    train_dataset = get_dataset(dataset_name, os.path.join(dataset_path, dataset_name) + '/train.csv')
    test_dataset = get_dataset(dataset_name, os.path.join(dataset_path, dataset_name) + '/test.csv')
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    LOGGER.info("Finished loading data")
    # 2. 根据数据集信息以及模型选择获取模型
    field_dims = train_dataset.field_dims
    numerical_num = train_dataset.numerical_num
    model = get_model(model_name, field_dims, numerical_num, task_num, expert_num, embed_dim).to(device)
    if 'compile' in dir(torch):
        LOGGER.info(f"{colorstr('Pytorch 2.0')} detected, compiling the model to speed up. ")
        model = torch.compile(model, mode="max-autotune")
        # model = torch.compile(model, mode="max-autotune", fullgraph=True, backend='Eager')
        # model = torch.compile(model, mode="max-autotune", fullgraph=True)
        LOGGER.info(f"{colorstr('Pytorch 2.0')} compilation done. ")
    
    # 3. 损失函数： TODO 根据多任务获得离散型和连续型的损失函数
    # 3.1离散型分类任务
    # 3.1.1普通分布
    # criterion = torch.nn.BCELoss()
    # 3.1.2类别不平衡
    criterion = AUCMLoss()
    # 3.2连续型回归任务
    # criterion = torch.nn.MSELoss()

    # 4. 优化器
    # 4.1 基础优化器
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # optimizer_sharedLayer = torch.optim.Adam(
    #     model.shared_parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer_taskLayer = torch.optim.Adam(
    #     model.specific_parameters(), lr=learning_rate, weight_decay=weight_decay)
    # # 多任务优化器
    # multitask_optimizer = MetaBalance(model.shared_parameters())

    # 5. 获得训练器
    trainer = get_trainer(model_name, model=model, 
                          optimizer=optimizer, data_loader=train_data_loader, 
                          criterion=criterion, device=device)

    early_stopper = EarlyStopper(kwargs.patience, kwargs.min_delta, kwargs.cumulative_delta)
    
    # 创建新的实验记录
    save_dir = Path(save_dir).resolve().absolute()
    i= 0
    while True:
        exp_save_dir = save_dir/f"exp_{i}"
        if not exp_save_dir.exists():
            exp_save_dir.mkdir(parents=True)
            save_dir = exp_save_dir
            break
        i+=1
    
    LOGGER.info("Begin to train")
    for epoch_i in range(epoch):
        trainer.train_epoch()
        auc, loss = test(model, test_data_loader, task_num, device)
        LOGGER.info('epoch:', epoch_i, 'test: auc:', auc)

        if epoch_i%args.save_epoch==0:
            save_path = save_dir / f"{model_name}_{epoch_i}.pt"
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
    LOGGER.info(colorstr('parameters: ') + ', '.join(f'{k}={v}' for k, v in args.items()))
    main(**args)
    
# %%
