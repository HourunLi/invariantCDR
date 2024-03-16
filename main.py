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
from invariantCDR.runner import Runner
import json



def create_arg_parser():
    """Create argument parser for our baseline. """
    parser = argparse.ArgumentParser('WSDM')

    # DATA  Arguments
    parser.add_argument('--domains', type=str, default="sport_cloth || electronic_cell, sport_cloth || game_video, uk_de_fr_ca_us", help='specify none ("none") or a few source markets ("-" seperated) to augment the data for training')
    parser.add_argument('--task', type=str, default='dual-user-intra', help='dual-user-intra, dual-user-inter, multi-item-intra, multi-user-intra')

    # MODEL Arguments
    parser.add_argument('--model', type=str, default='UniCDR', help='right model name')
    parser.add_argument('--mask_rate', type=float, default=0.1, help='mask rate of interactions')
    parser.add_argument('--num_epoch', type=int, default=100, help='number of epoches')
    parser.add_argument('--aggregator', type=str, default='mean', help='switching the user-item aggregation')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam',
                        help='Optimizer: sgd, adagrad, adam or adamax.')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--l2_reg', type=float, default=1e-7, help='the L2 weight')
    parser.add_argument('--lr_decay', type=float, default=0.98, help='decay learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='decay learning rate')
    parser.add_argument('--latent_dim', type=int, default=128, help='latent dimensions')
    parser.add_argument('--num_negative', type=int, default=10, help='num of negative samples during training')
    parser.add_argument('--maxlen', type=int, default=10, help='num of item sequence')
    parser.add_argument('--dropout', type=float, default=0.3, help='random drop out rate')
    parser.add_argument('--save', action='store_true', help='save model?')
    parser.add_argument('--lambda', type=float, default=50, help='the parameter of EASE')
    parser.add_argument('--lambda_a', type=float, default=0.5, help='for our aggregators')
    parser.add_argument('--lambda_loss', type=float, default=0.4, help='the parameter of loss function')
    parser.add_argument('--static_sample', action='store_true', help='accelerate the dataloader')

    # others
    parser.add_argument('--cuda', action='store_true', help='use of cuda')
    parser.add_argument('--seed', type=int, default=42, help='manual seed init')
    parser.add_argument('--decay_epoch', type=int, default=10, help='Decay learning rate after this epoch.')

    return parser


def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    opt = vars(args)
    opt["device"] = torch.device('cuda' if torch.cuda.is_available() and opt["cuda"] else 'cpu')
    print_config(opt)
    args, data = load_data(args)

    runner = Runner(args, model, data)
    results = []

    if args.mode == "train":
        results = runner.run()
    elif args.mode == "eval":
        results = runner.re_run()

    # post-logs
    measure_dict = results
    info_dict.update(measure_dict)
    filename = "info_" + args.dataset + ".json"
    json.dump(info_dict, open(osp.join(log_dir, filename), "w"))




if __name__ == "__main__":
    main()

