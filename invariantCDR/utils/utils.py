import random
import numpy as np
import torch 

def print_config(config):
    info = "Running with the following configs:\n"
    for k, v in config.items():
        info += "\t{} : {}\n".format(k, str(v))
    print("\n" + info + "\n")

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_user_idx(user, user_base = 0):
    return 2 * (user + user_base)
    
def get_item_idx(item, item_base = 0):
    return 2 * (item + item_base) + 1