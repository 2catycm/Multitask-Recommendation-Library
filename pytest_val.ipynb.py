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
from callbacks.debug_step_callback import JustTestCanRun
from tqdm import tqdm
callback = JustTestCanRun(True)
callback = JustTestCanRun(False)
for i in tqdm(range(10000)):
    
    stop_iteration = False
    for c in [callback]:
        try:
            callback(i)
        except StopIteration:
            stop_iteration = True
    if stop_iteration:
        break

# %%
