import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import argparse
import os
import os.path as osp
import shutil
import numpy as np
import pickle
import yaml
from copy import deepcopy
from itertools import repeat
from torch_geometric.data import DataLoader, InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data
from torch_geometric.utils import degree
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, global_max_pool
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

from utils.utils import *
from invariantCDR.model import *
from invariantCDR.data import *
from invariantCDR.runner import *
from invariantCDR.config import args
import json


def main():
    args, data = load_data(args)
    print_config(args)

    # model = invariantCDR(args = args)
    # runner = Runner(args, data, model)
    # results = []

    # if args.mode == "train":
    #     results = runner.run()
    # elif args.mode == "eval":
    #     results = runner.re_run()


if __name__ == "__main__":
    main()

