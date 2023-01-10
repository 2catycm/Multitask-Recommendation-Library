#%%
from pathlib import Path
this_file = Path(__file__).resolve().absolute()
this_directory = this_file.parent
project_directory = this_directory.parent
import sys
if str(project_directory) not in sys.path:
    sys.path.append(str(project_directory))
    
    

def get_multitask_optimizers():
    pass


#%%

