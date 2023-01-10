import wandb
import train
def sweep():
    wandb.init()
    # Get hyp dict from sweep agent. Copy because train() modifies parameters which confused wandb.
    # params = vars(wandb.config).get("_items").copy()
    train.main(wandb.config)

if __name__ == "__main__":
    sweep()
