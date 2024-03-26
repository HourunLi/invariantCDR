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
    
    
def move2GPU(data, device):
    data["x"] = torch.tensor(data["x"]).to(device)
    data["train"]["edge_list"] = torch.Tensor(data["train"]["edge_list"]).to(device)
    return
    