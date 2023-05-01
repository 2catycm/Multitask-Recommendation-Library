import torch
from models.layers import EmbeddingLayer, MultiLayerPerceptron

from models.abstract_multitask_model import MultitaskModel


class SingleTaskModel(MultitaskModel):
    """
    A pytorch implementation of Single Task Model.
    """

    def __init__(self, categorical_field_dims, numerical_num, task_num,
                 embed_dim, bottom_mlp_dims, tower_mlp_dims, dropout
                 ,*args, **kwargs):
        super().__init__(categorical_field_dims, numerical_num, task_num,
                         embed_dim, bottom_mlp_dims, tower_mlp_dims, dropout)
        # 共享的特征预处理层
        self.embedding = torch.nn.ModuleList(
            [EmbeddingLayer(categorical_field_dims, embed_dim) for i in range(task_num)])
        self.numerical_layer = torch.nn.ModuleList(
            [torch.nn.Linear(numerical_num, embed_dim) for i in range(task_num)])
        self.embed_output_dim = (len(categorical_field_dims) + 1) * embed_dim
        # 底层网络和上层网络
        self.bottom = torch.nn.ModuleList([MultiLayerPerceptron(
            self.embed_output_dim, bottom_mlp_dims, dropout, output_layer=False) for i in range(task_num)])
        self.tower = torch.nn.ModuleList([MultiLayerPerceptron(
            bottom_mlp_dims[-1], tower_mlp_dims, dropout) for i in range(task_num)])

    def forward(self, categorical_x, numerical_x):
        results = list()
        for i in range(self.task_num):
            categorical_emb = self.embedding[i](categorical_x)
            numerical_emb = self.numerical_layer[i](numerical_x).unsqueeze(1)
            emb = torch.cat([categorical_emb, numerical_emb],
                            1).view(-1, self.embed_output_dim)
            fea = self.bottom[i](emb)
            results.append(torch.sigmoid(self.tower[i](fea).squeeze(1)))
        return results
    
    def specific_parameters(self):
        # 上层和下层网络网络都是分离的。
        return self.parameters_selected([self.bottom, self.tower])

    def shared_parameters(self):
        # 只是共享特征预处理层
        return self.parameters_selected([self.embedding, self.numerical_layer])