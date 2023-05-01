from abc import ABC, abstractmethod
import torch
from typing import List


    
class MultitaskModel(torch.nn.Module):
    def __init__(self, categorical_field_dims: List[int], numerical_num: int,
                 task_num: int,
                 embed_dim: int, bottom_mlp_dims: List[int], tower_mlp_dims: List[int],
                 dropout:float, 
                 *args, **kwargs) -> None:
        """需要指定输入的模式、输出的模式、以及中间模型的参数

        Args:
            categorical_field_dims (List[int]): 输入的离散特征的取值上界（取不到）
            numerical_num (int): 数值特征的数量
            task_num (int): 任务数量，即输出维度
            embed_dim (int): 模型参数：嵌入层的维度
            bottom_mlp_dims (List[int]): 下层网络的感知机隐藏层特征数量
            tower_mlp_dims (List[int]): 上层高塔网络的感知机隐藏层特征数量
            dropout(float): dropout
        """
        super().__init__(*args, **kwargs)
        self.categorical_field_dims = categorical_field_dims
        self.numerical_num = numerical_num
        self.task_num = task_num
        self.embed_dim = embed_dim
        self.bottom_mlp_dims = bottom_mlp_dims
        self.tower_mlp_dims = tower_mlp_dims
        self.dropout = dropout
    @abstractmethod
    def forward(self, categorical_x, numerical_x):
        """
        :param 
        categorical_x: Long tensor of size ``(batch_size, categorical_field_dims)``
        numerical_x: Long tensor of size ``(batch_size, numerical_num)``
        """
        pass
    # def forward(self, X):
    #     """兼容Pytorch的forward

    #     Args:
    #         X (torch.Tensor): _description_
    #     """

    def parameters_selected(self, selected):
        result = []
        for i in selected:
            result += list(i.parameters())
        return iter(result)

    @abstractmethod
    def specific_parameters(self):
        pass

    @abstractmethod
    def shared_parameters(self):
        pass


class MultitaskWrapper(torch.nn.Module):
    def __init__(self, m:MultitaskModel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.m = m
    def forward(self, X:torch.Tensor):
        return self.m(X[:, :len(self.m.categorical_field_dims)].to(torch.int),
                 X[:, -self.m.numerical_num:])