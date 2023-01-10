from abc import ABC, abstractmethod
import numpy as np
import torch.utils.data

class MultitaskDataset(torch.utils.data.Dataset):
    def set_param(self, categorical_data, numerical_data, labels, labels_type=None, categorical_num=None, 
                 numerical_num=None, labels_num=None, field_dims=None) -> None:
        """多任务学习典型的数据集。
        为了让本项目内的模型和训练代码正常工作，不出bug，我们提供这个超类，要求数据集至少有以下属性：
        Args:
            categorical_data (np.ndarray):  NxM_c 输入的离散型随机变量数据。
            numerical_data (np.ndarray):    NxM_n 输入的连续型随机变量数据。
            labels (np.ndarray):            NxQ 标签数据，可能是离散型或者连续型。推荐系统中可能也有回归任务，比如腾讯PLE论文中的任务或MMOE任务。
            labels_type (np.ndarray):       Q 标记labels的类型，0表示离散型，1表示连续型。默认为0。离散型即分类任务，比如CTR估计任务。
        而以下属性是可选的, 可以自动推导。
        Args:
            categorical_num (int):          M_c 离散型随机变量的数量。
            numerical_num (int):            M_n 连续型随机变量的数量。
            labels_num (int):               Q 标签的数量。
            field_dims (np.ndarray):        M_c 离散型随机变量的取值范围。
        """
        # 必选项
        self.categorical_data = categorical_data
        self.numerical_data = numerical_data
        self.labels = labels
        self.labels_type = labels_type if labels_type is not None else np.zeros(labels.shape[1])
        # 可选项
        self.categorical_num = categorical_num if categorical_num is not None else categorical_data.shape[1]
        self.numerical_num = numerical_num if numerical_num is not None else numerical_data.shape[1]
        self.labels_num = labels_num if labels_num is not None else labels.shape[1]
        self.field_dims = field_dims if field_dims is not None else np.max(categorical_data, axis=0) + 1
        
    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return self.categorical_data[index], self.numerical_data[index], self.labels[index]
    