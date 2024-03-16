
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
from torch.nn import functional as F
from torch import nn
from tqdm import tqdm
from utils.utils import *
import os
import sys
import time
import torch
import numpy as np
import torch.optim as optim
import pandas as pd
import math

class Runner(object):
    def __init__(self, args, model, data, writer=None, **kwargs):
        seed_everything(args.seed)
        self.args = args
        self.data = data
        self.model = model
        self.len = len(data["train"]["edge_list"]) # number of graphs

    def train(self, epoch, data):
        args = self.args
        self.model.train()
        optimizer = self.optimizer
        
    def run(self):
        args = self.args
        min_epoch = args.min_epoch
        max_patience = args.patience
        patience = 0
        self.model.train()
        optimizer = self.optimizer
        emb, y = self.model.encoder.get_embeddings(
            [self.data["train"]["edge_lists"][ind].long().to(args.device) for ind in range(self.len)]
        )