#%%
from pathlib import Path
this_file = Path(__file__).resolve().absolute()
this_directory = this_file.parent
project_directory = this_directory.parent
import sys
if str(project_directory) not in sys.path:
    sys.path.append(str(project_directory))

#%%
from torch.optim import SGD, Adam, AdamW 
from libauc.optimizers import PESG
   
def get_optimizer(optimizer_name, model_weights, **kwargs):
    lower_name = optimizer_name.lower()
    return {
        'sgd':SGD(model_weights, **kwargs),
        'adam':Adam(model_weights, **kwargs),
        'adamw':AdamW(model_weights, **kwargs),
        'pesg': PESG(model_weights, **kwargs),
    }[lower_name]
    # if lower_name=='sgd':
    #     return SGD(model_weights, **kwargs)
    # elif lower_name==

#%%

