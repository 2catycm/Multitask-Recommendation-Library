import torch
from models.layers import EmbeddingLayer, MultiLayerPerceptron
from models.abstract_multitask_model import MultitaskModel


class DensePLEModel(MultitaskModel):
    """
    A pytorch implementation of PLE Model.

    Reference:
        Tang, Hongyan, et al. Progressive layered extraction (ple): A novel multi-task learning (mtl) model for personalized recommendations. RecSys 2020.
    """

    def __init__(self, categorical_field_dims, numerical_num, task_num,
                 embed_dim, bottom_mlp_dims, tower_mlp_dims, dropout,
                 shared_expert_num, specific_expert_num
                 ,*args, **kwargs):
        super().__init__(categorical_field_dims, numerical_num, task_num,
                         embed_dim, bottom_mlp_dims, tower_mlp_dims, dropout)
        # 共享的特征预处理层
        self.embedding = EmbeddingLayer(categorical_field_dims, embed_dim)
        self.numerical_layer = torch.nn.Linear(numerical_num, embed_dim)
        self.embed_output_dim = (len(categorical_field_dims) + 1) * embed_dim
        # 底层网络
        self.shared_expert_num = shared_expert_num
        self.specific_expert_num = specific_expert_num
        self.layers_num = len(bottom_mlp_dims)
        # self.layers_num = layers_num
        # 每个任务有自己的expert和gate
        self.task_experts = [
            [None] * self.task_num for _ in range(self.layers_num)]
        self.task_gates = [
            [None] * self.task_num for _ in range(self.layers_num)]
        # 中间大家一起共享了一些expert和gate
        self.share_experts = [None] * self.layers_num
        self.share_gates = [None] * self.layers_num
        
        # cumulative_output_dim = self.embed_output_dim
        cumulative_output_dim = self.bottom_mlp_dims[0]
        for i in range(self.layers_num):
            # 构建每一层的所有expert和gate
            # input_dim = self.embed_output_dim if 0 == i else bottom_mlp_dims[i - 1]
            
            # 前面所有输出的拼接是这一层的输入。这一层算完之后，也要把输出拼接到结果上去。
            input_dim = cumulative_output_dim
            output_dim = bottom_mlp_dims[i]
            cumulative_output_dim += output_dim
            
            self.share_experts[i] = torch.nn.ModuleList(
                [MultiLayerPerceptron(input_dim, [output_dim], 
                                      dropout, output_layer=False) 
                    for k in range(self.shared_expert_num)])
            self.share_gates[i] = torch.nn.Sequential(torch.nn.Linear(
                input_dim, shared_expert_num + task_num * specific_expert_num), torch.nn.Softmax(dim=1))
            for j in range(task_num):
                self.task_experts[i][j] = torch.nn.ModuleList(
                    [MultiLayerPerceptron(input_dim, [output_dim], 
                                          dropout, output_layer=False) 
                     for k in range(self.specific_expert_num)])
                self.task_gates[i][j] = torch.nn.Sequential(torch.nn.Linear(
                    input_dim, shared_expert_num + specific_expert_num), torch.nn.Softmax(dim=1)) # 只有专业专家集合+share专家集合
            self.task_experts[i] = torch.nn.ModuleList(self.task_experts[i])
            self.task_gates[i] = torch.nn.ModuleList(self.task_gates[i])

        self.task_experts = torch.nn.ModuleList(self.task_experts)
        self.task_gates = torch.nn.ModuleList(self.task_gates)
        self.share_experts = torch.nn.ModuleList(self.share_experts)
        self.share_gates = torch.nn.ModuleList(self.share_gates)

        # 上层网络
        # self.tower = torch.nn.ModuleList([MultiLayerPerceptron(
        #     bottom_mlp_dims[-1], tower_mlp_dims, dropout) for i in range(task_num)])
        self.tower = torch.nn.ModuleList([MultiLayerPerceptron(
            cumulative_output_dim, tower_mlp_dims, dropout) for i in range(task_num)])
        
        # emb太大了，缩小一点
        self.emb_downsampler = MultiLayerPerceptron(self.embed_output_dim, 
                                               [self.bottom_mlp_dims[0]], 
                                               dropout, output_layer=False)

    def forward(self, categorical_x, numerical_x):
        categorical_emb = self.embedding(categorical_x)
        numerical_emb = self.numerical_layer(numerical_x).unsqueeze(1)
        emb = torch.cat([categorical_emb, numerical_emb],
                        1).view(-1, self.embed_output_dim) # batch size, output_dim
        emb = self.emb_downsampler(emb)
        # print(emb.shape)
        # print(categorical_emb.shape)
        # print(numerical_emb.shape)
        # task1 input ,task2 input,..taskn input, share_expert input
        task_fea = [emb for i in range(self.task_num + 1)] # 为他们准备输入
        for i in range(self.layers_num):
            share_output = [expert(task_fea[-1]).unsqueeze(1)
                            for expert in self.share_experts[i]]  
            task_output_list = []
            for j in range(self.task_num):
                task_output = [expert(task_fea[j]).unsqueeze(1)
                               for expert in self.task_experts[i][j]]
                task_output_list.extend(task_output)
                mix_ouput = torch.cat(task_output+share_output, dim=1)
                gate_value = self.task_gates[i][j](task_fea[j]).unsqueeze(1)
                current_output = torch.bmm(gate_value, mix_ouput).squeeze(1)
                task_fea[j] = torch.hstack((current_output,task_fea[j]))
            # 如果不是最后一层，需要计算share expert 的输出
            if i != self.layers_num-1:  # not last layer
                gate_value = self.share_gates[i](task_fea[-1]).unsqueeze(1)
                mix_ouput = torch.cat(task_output_list + share_output, dim=1)
                current_output = torch.bmm(gate_value, mix_ouput).squeeze(1)
                task_fea[-1] = torch.hstack((current_output, task_fea[-1]))

        results = [torch.sigmoid(self.tower[i](task_fea[i]).squeeze(1))
                   for i in range(self.task_num)]
        return results

    def specific_parameters(self):
        return self.parameters_selected([self.task_experts, self.tower, self.task_gates])

    def shared_parameters(self):
        return self.parameters_selected([self.share_experts, self.share_gates, self.embedding, self.numerical_layer])
