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

def loadFile(opt, mode):
    edge_list = []
    for _, cur_domain in enumerate(opt["domains"]):
        cur_src_data_dir = os.path.join("../datasets/"+str(opt["task"]) + "/dataset/", cur_domain + f"/{mode}.txt")
        print(f'Loading {cur_domain}: {cur_src_data_dir}')
        edges = []
        with codecs.open(cur_src_data_dir, "r", encoding="utf-8") as infile:
            for line in infile:
                line = line.strip().split("\t")
                user = int(line[0])
                item = int(line[1]) + 1
                edges.append([user, item])
        edge_list.append(edges)
    edge_list = edge_list.transpose(0, 1)
    print(len(edge_list), len(edge_list[0]))
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
    data["valid"]["edge_lists"] = loadFile(args, mode = "valid")
    data["test"]["edge_lists"] = loadFile(args, mode = "test")
    
    # set args
    args.graphs = len(args.domains)
    return args, data