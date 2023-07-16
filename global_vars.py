tensorboard = None
wandb = False
clearml = False


# from accelerate import Accelerator
# from accelerate.utils import LoggerType

# accelerator = Accelerator(split_batches=True,
#                         #   log_with=[LoggerType.WANDB, LoggerType.TENSORBOARD
#                         #             # , LoggerType.COMETML
#                         #             ],
#                         #   logging_dir="./tensorboard"
#                           ) # batch_size 始终由用户控制，不随GPU数量变化



# def print_main_process(*args, **kwargs):
#     if accelerator.is_main_process:
#         print(*args, **kwargs)