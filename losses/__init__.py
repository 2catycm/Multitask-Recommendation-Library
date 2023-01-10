#%%
from pathlib import Path
this_file = Path(__file__).resolve().absolute()
this_directory = this_file.parent
project_directory = this_directory.parent
import sys
if str(project_directory) not in sys.path:
    sys.path.append(str(project_directory))
#%%
from libauc.losses import AUCMLoss
from torch.nn import BCELoss, MSELoss

def get_loss(loss_name):
    lower_name = loss_name.lower()
    return {
        'bceloss':BCELoss(),
        'mseloss':MSELoss(),
        'aucmloss':AUCMLoss(),
    }[lower_name]