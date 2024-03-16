import os
import torch
import random
import resource
# import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
import pdb
from collections import defaultdict
import pickle
import codecs

def loadFile(args, mode):
    edge_list = []
    args.user_max, args.item_max = [], []
    for _, cur_domain in enumerate(args.domains):
        cur_src_data_dir = os.path.join("./datasets/"+str(args.task) + "/dataset/", cur_domain + f"/{mode}.txt")
        print(f'Loading {cur_domain}: {cur_src_data_dir}')
        edges = []
        max_user, max_item = 0, 0
        with codecs.open(cur_src_data_dir, "r", encoding="utf-8") as infile:
            for line in infile:
                line = line.strip().split("\t")
                user = int(line[0])
                item = int(line[1]) + 1
                max_user = max(max_user, user)
                max_item = max(max_item, item)
                edges.append([user, item])
        edges = torch.tensor(edges).to(args.device).transpose(0, 1)
        edge_list.append(edges)
        args.user_max.append(max_user + 1)
        args.item_max.append(max_item + 1)
    # for i in range(len(edge_list)):
    #     print(edge_list[i].size())
    return edge_list

def load_data(args):
    data = defaultdict(dict)
    if "dual" in args.task:
        filename = args.domains.split("_")
        args.domains = []
        args.domains.append(filename[0] + "_" + filename[1])
        args.domains.append(filename[1] + "_" + filename[0])
    else:
        args.domains = args.domains.split('_')
        
    print("Loading domains:", args.domains)
    data["train"]["edge_lists"] = loadFile(args, mode = "train")
    # data["valid"]["edge_lists"] = loadFile(args, mode = "valid")
    data["test"]["edge_lists"] = loadFile(args, mode = "test")
    
    # set args
    args.num_domains = len(args.domains)
    args.
            #     self.user_embedding_list.append(nn.Embedding(args.user_max[i], args.latent_dim))
            # self.item_embedding_lsit.append(nn.Embedding(args.item_max[i] + 1, args.latent_dim, padding_idx=0))
    return args, data