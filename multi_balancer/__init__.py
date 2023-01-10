#%%
from pathlib import Path
this_file = Path(__file__).resolve().absolute()
this_directory = this_file.parent
project_directory = this_directory.parent
import sys
if str(project_directory) not in sys.path:
    sys.path.append(str(project_directory))
    
#%%    
from multi_balancer.metabalance import *
def get_multi_balancer(balancer_name, shared_params, relax_factor=0.7, beta=0.9):
    lower_name = balancer_name.lower()
    return {
        'metabalance':MetaBalance(shared_params, relax_factor, beta),
    }[lower_name]

#%%

