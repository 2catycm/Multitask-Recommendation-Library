#%%
import torch
zeros = torch.zeros(4, 2)
targets = torch.randint(0, 2, (4,2 ))
zeros.scatter_(1, targets, 1)
# %%
zeros
# %%
