# %%
from models.abstract_multitask_model import MultitaskModel, MultitaskWrapper
from models.metaheac import MetaHeacModel
from models.aitm import AITMModel
from models.ple import PLEModel
from models.mmoe import MMoEModel
from models.omoe import OMoEModel
from models.singletask import SingleTaskModel
from models.sharedbottom import SharedBottomModel
import sys
from pathlib import Path
this_file = Path(__file__).resolve().absolute()
this_directory = this_file.parent
project_directory = this_directory.parent
if str(project_directory) not in sys.path:
    sys.path.append(str(project_directory))

# %%

# def get_model(name, categorical_field_dims, numerical_num, task_num, expert_num, embed_dim)->MultitaskModel:

model_names = ['sharedbottom', 'singletask',
               'omoe', 'mmoe', 'ple', 'aitm', 'metaheac']


def get_model(name, categorical_field_dims, numerical_num, task_num,
              expert_num, embed_dim,
              bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64),
              dropout=0.2, *args, **kwargs):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    if name == 'sharedbottom':
        print("Model: Shared-Bottom")
        return SharedBottomModel(categorical_field_dims, numerical_num, task_num,
                                 embed_dim=embed_dim, bottom_mlp_dims=bottom_mlp_dims, tower_mlp_dims=tower_mlp_dims,
                                 dropout=dropout, *args, **kwargs)
    elif name == 'singletask':
        print("Model: SingleTask")
        return SingleTaskModel(categorical_field_dims, numerical_num, task_num,
                               embed_dim=embed_dim, bottom_mlp_dims=bottom_mlp_dims, tower_mlp_dims=tower_mlp_dims,
                               dropout=dropout, *args, **kwargs)
    elif name == 'omoe':
        print("Model: OMoE")
        return OMoEModel(categorical_field_dims, numerical_num, task_num,
                         embed_dim=embed_dim, bottom_mlp_dims=bottom_mlp_dims, tower_mlp_dims=tower_mlp_dims,
                         dropout=dropout,
                         expert_num=expert_num, *args, **kwargs)
    elif name == 'mmoe':
        print("Model: MMoE")
        return MMoEModel(categorical_field_dims, numerical_num, task_num,
                         embed_dim=embed_dim, bottom_mlp_dims=bottom_mlp_dims, tower_mlp_dims=tower_mlp_dims,
                         dropout=dropout,
                         expert_num=expert_num, *args, **kwargs)
    elif name == 'ple':
        print("Model: PLE")
        return PLEModel(categorical_field_dims, numerical_num, task_num,
                        embed_dim=embed_dim, bottom_mlp_dims=bottom_mlp_dims, tower_mlp_dims=tower_mlp_dims,
                        dropout=dropout,
                        shared_expert_num=int(expert_num / 2), specific_expert_num=int(expert_num / 2), *args, **kwargs)
    elif name == 'aitm':
        print("Model: AITM")
        return AITMModel(categorical_field_dims, numerical_num, task_num,
                         embed_dim=embed_dim, bottom_mlp_dims=bottom_mlp_dims, tower_mlp_dims=tower_mlp_dims,
                         dropout=dropout, *args, **kwargs)
    elif name == 'metaheac':
        print("Model: MetaHeac")
        return MetaHeacModel(categorical_field_dims, numerical_num, task_num,
                             embed_dim=embed_dim, bottom_mlp_dims=bottom_mlp_dims, tower_mlp_dims=tower_mlp_dims,
                             dropout=dropout,
                             expert_num=expert_num, critic_num=5, *args, **kwargs)
    else:
        raise ValueError('unknown model name: ' + name)


# %%
