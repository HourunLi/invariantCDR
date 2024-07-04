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

from model.invariantCDR import *
from data import load_data
from runner import *
from config import args
import json

# random.seed(0)
# torch.manual_seed(42)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(42)

if int(args.device_id) >= 0 and torch.cuda.is_available():
    args.device = torch.device("cuda:{}".format(args.device_id))
    print("using gpu:{} to train the model".format(args.device_id))
else:
    args.device = torch.device("cpu")
    print("using cpu to train the model")
device = args.device
print(device)
print(args)

args.feature_dim = int(args.feature_dim // args.num_latent_factors) * args.num_latent_factors
args, train_batch, source_valid_batch, source_test_batch, target_valid_batch, target_test_batch, source_dev_batch, target_dev_batch = load_data(args)
recmodel = invariantCDR(args, device)
optimizer = torch_utils.get_optimizer(args.optim, recmodel.parameters(), args.lr, args.weight_decay)
start_epoch = 1
args.model_save_dir = args.save_dir + '/' + args.model_name
if args.load:
    helper.ensure_dir(args.model_save_dir, verbose=True)
    model_path = args.model_save_dir + '/' + args.model_file
    checkpoint = torch.load(model_path)
    recmodel.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print("Loading model from {}".format(model_path))
runner = Runner(args, recmodel, train_batch, source_valid_batch, source_test_batch, target_valid_batch, target_test_batch, source_dev_batch, target_dev_batch, start_epoch = start_epoch)

torch.set_printoptions(threshold=np.inf)
if args.mode == "train":
    results = runner.train()
# elif args.mode == "eval":
#     results = runner.eval()
