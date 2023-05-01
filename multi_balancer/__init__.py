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
from multi_balancer.corrbalance import *
def get_multi_balancer(balancer_name, shared_params, corr_factor=None, relax_factor=0.7, beta=0.9):
    lower_name = balancer_name.lower()
    if corr_factor == None:
        corr_factor = []
    # shared_params = list(shared_params)
    if lower_name == 'metabalance':
        return MetaBalance(shared_params, relax_factor, beta)
    elif lower_name == 'corrbalance':
        return CorrBalance(shared_params, corr_factor, relax_factor, beta)


#%%

