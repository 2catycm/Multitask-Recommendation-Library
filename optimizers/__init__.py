# %%
from torch.optim import SGD, Adam, AdamW
from libauc.optimizers import PESG
import sys
from pathlib import Path
this_file = Path(__file__).resolve().absolute()
this_directory = this_file.parent
project_directory = this_directory.parent
if str(project_directory) not in sys.path:
    sys.path.append(str(project_directory))

# %%


def get_optimizer0(optimizer_name, model_weights, **kwargs):
    lower_name = optimizer_name.lower()

    if lower_name == 'sgd':
        return SGD(model_weights, kwargs["lr"], weight_decay=kwargs.get("weight_decay", 0), momentum=kwargs.get("momentum", 0))
    elif lower_name == 'adam':
        return Adam(model_weights, kwargs["lr"], weight_decay=kwargs.get("weight_decay", 0))
    elif lower_name == 'adamw':
        return AdamW(model_weights, kwargs["lr"], weight_decay=kwargs.get("weight_decay", 0))


def get_optimizer(optimizer_name, model_weights, model, criti, device, **kwargs):
    lower_name = optimizer_name.lower()
    if lower_name == 'pesg':
        return PESG(model, loss_fn=criti, lr=kwargs["lr"],
                    margin=kwargs.get("margin", 1.0),
                    weight_decay=kwargs.get("weight_decay", 0.0001),
                    epoch_decay=kwargs.get("epoch_decay ", 0.003),
                    momentum=kwargs.get("momentum", 0.9),
                    device=device
                    )
    else:
        return get_optimizer0(optimizer_name, model_weights, **kwargs)


# %%
