#%%
from train import select_device
d = select_device('auto')
# select_device(1)
import torch
a = torch.zeros((4, 2))
a.to(d)
#%%
# from models import get_model
# model = get_model('ple', [0,1,2], 3, 2, 8, 3)
# %%

import torch

criti = torch.nn.BCELoss()

input = torch.rand((4, 2))
input
target = torch.zeros((4, 2))+0.5
target
criti(input, target)


input = torch.rand((4, 2))


# %%
# criti(input, target.squeeze(-1))

# %%
# from pytext.loss.loss import AUCPRHingeLoss
from losses.AUCPRHingeLoss import AUCPRHingeLoss

criti = AUCPRHingeLoss(num_classes=2)
# input = torch.rand((4, 2))
# input = torch.zeros((4, 2))
# input = torch.ones((4, 2))
# target = torch.zeros((4, 2))+0.5
# target = torch.zeros((4, 1)).long()
# target = torch.ones((4, 2)).long()

target = torch.Tensor([0, 0, 1, 1])
# input = torch.Tensor([0.9, 0.9, 0.8, 0.7])
input = torch.Tensor([0, 0, 1, 1])
input = torch.vstack([input, 1-input]).T
target = target.unsqueeze(-1).long()
input
#%%
criti(input, target)
#%%
import torch
from losses.AUCPRHingeLoss import AUCPRHingeLoss
criti = AUCPRHingeLoss(num_classes=2)
def criti_it(i, t, criti=criti, unsqueeze=False):
    if unsqueeze:
        t = torch.Tensor(t).unsqueeze(-1).long()
    else:
        t = torch.Tensor(t).long()
    i = torch.Tensor(i)
    i = torch.vstack([i, 1-i]).T
    return criti(i, t)
criti_it([0, 0, 1, 1], [0, 0, 1, 1])
# criti_it([1, 1, 0, 0], [1, 1, 0, 0])
# criti_it([1, 1, 1, 1], [1, 1, 0, 0])
# criti_it([1, 1, 1, 1], [1, 1, 1, 1])
# criti_it([1, 1, 1, 1], [0, 0, 0, 0])
#%%
criti_it([0.001]*10000, torch.randint(0, 2, (10000,)))
target = torch.zeros(10000)
target[0:5] = 1
criti_it([0.001]*10000, target)

from sklearn.metrics import roc_auc_score
roc_auc_score(target, [0.001]*10000) # Only one class present in y_true. ROC AUC score is not defined in that case.
# roc_auc_score([0]*10000, [0.001]*10000)
#%%
import torch
def transf(i, t,  unsqueeze=False):
    t = torch.Tensor(t)
    t = torch.vstack([t, 1-t]).T
    i = torch.Tensor(i)
    i = torch.vstack([i, 1-i]).T
    return i, t
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
criti = AUCMLoss()
criti_it = lambda i, t:criti(i.cuda(), t.cuda())
# criti_it(*transf([0.001]*10000, target, True))
# criti_it(*transf([0.001]*10000, [0]*10000, True))
# criti_it(*transf([0.001]*10000, [0.2]*10000, True))
i, t = transf([0.001]*10000, [0.2]*10000, True)
i,t
criti_it(*transf([0.001]*10000, [1]*10000, True))

# %%
def fun(a, b):
    print(a, b)
d = {'a': 1, 'b': 2}
fun(**d)
d = {'a': 1, 'b': 2, 'c': 3}
fun(**d) # 报错
# %%
def fun2(a, b, **kargs):
    print(a, b)
    print(kargs)
d = {'a': 1, 'b': 2, 'c': 3}
fun2(**d) # 没问题
    
