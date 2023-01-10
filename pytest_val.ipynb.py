# from ignite.handlers import EarlyStopping
#%%
from utils.general import LOGGER

LOGGER.info("hello AI")

# %%
from utils.torch_utils import make_exp_reproducible
make_exp_reproducible()
# %%
import torch
torch.compile
# %%
