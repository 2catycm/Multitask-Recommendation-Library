from abc import ABC, abstractmethod
import torch

class MultitaskModel(torch.nn.Module):
    def parameters_selected(self, selected):
        result = []
        for i in selected:
            result += list(i.parameters())
        return result
    @abstractmethod
    def specific_parameters(self):
        pass
    
    @abstractmethod
    def shared_parameters(self):
        pass