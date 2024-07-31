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
import json
import pandas as pd
import numpy as np
import torch.nn as nn
from utils.GraphMaker import GraphMaker
from utils.loader import DataLoader

def generate_train_dev(train_batch, source_shared_user, target_shared_user, filenames):
    filter_source_data = [item for item in train_batch.source_train_data if item[0] >= source_shared_user]
    filter_target_data = [item for item in train_batch.target_train_data if item[0] >= target_shared_user]

    sampled_source_data = random.sample(filter_source_data, 1000)
    sampled_target_data = random.sample(filter_target_data, 1000)

    source_path = "../dataset/" + filenames[0] + "_" + filenames[1] + "/dev.txt"
    with open(source_path, 'w') as source_file:
        for item in sampled_source_data:
            source_file.write(f"{item[0]}\t{item[1]}\n")
            
    target_path = "../dataset/" + filenames[1] + "_" + filenames[0] + "/dev.txt"
    with open(target_path, 'w') as target_file:
        for item in sampled_target_data:
            target_file.write(f"{item[0]}\t{item[1]}\n")
    return sampled_source_data, sampled_target_data

def load_data(args):
    filenames = args.domains.split("_")
    source_train_data = "../dataset/" + filenames[0] + "_" + filenames[1] + "/train.txt"
    source_G = GraphMaker(args, source_train_data)
    args.source_G = source_G
    args.source_UV = source_G.UV
    args.source_VU = source_G.VU
    args.source_adj = source_G.adj
    
    target_train_data = "../dataset/" + filenames[1] + "_" + filenames[0] + "/train.txt"
    target_G = GraphMaker(args, target_train_data)
    args.target_G = target_G
    args.target_UV = target_G.UV
    args.target_VU = target_G.VU
    args.target_adj = target_G.adj
    print("graph loaded!")
    
    print("Loading data from {} with batch size {}...".format(args.domains, args.batch_size))
    train_batch = DataLoader(filenames, args.batch_size, args, evaluation = -1)
    source_valid_batch = DataLoader(filenames, args.batch_size, args, evaluation = 3)
    source_test_batch = DataLoader(filenames, args.batch_size, args, evaluation = 1)
    target_valid_batch = DataLoader(filenames, args.batch_size, args, evaluation = 4)
    target_test_batch = DataLoader(filenames, args.batch_size, args, evaluation = 2)

    print("source_user_num", args.source_user_num)
    print("target_user_num", args.target_user_num)
    print("source_item_num", args.source_item_num)
    print("target_item_num", args.target_item_num)
    args.shared_user = min(source_test_batch.MIN_USER, target_test_batch.MIN_USER) + 1
    args.test_user = max(source_test_batch.MAX_USER, target_test_batch.MAX_USER) + 1
    args.source_shared_user = source_test_batch.MAX_USER + 1
    args.target_shared_user = target_test_batch.MAX_USER + 1
    print("shared users id: " + str(args.shared_user))
    print("test users {}, {}".format(source_test_batch.MAX_USER - source_test_batch.MIN_USER + 1 , target_test_batch.MAX_USER - target_test_batch.MIN_USER + 1))
    if not os.path.exists("../dataset/" + filenames[0] + "_" + filenames[1] + "/dev.txt"):
        generate_train_dev(train_batch, args.source_shared_user, args.target_shared_user, filenames)
    source_dev_batch = DataLoader(filenames, args.batch_size, args, evaluation = 5)
    target_dev_batch = DataLoader(filenames, args.batch_size, args, evaluation = 6)
    print(f"source train data : {len(train_batch.source_train_data)}, target train data: {len(train_batch.target_train_data)}, source test data : {len(source_test_batch.test_data)}, target test data : {len(target_test_batch.test_data)}, source dev data : {len(source_dev_batch.test_data)}, target dev data : {len(target_dev_batch.test_data)}")
    return args, train_batch, source_valid_batch, source_test_batch, target_valid_batch, target_test_batch, source_dev_batch, target_dev_batch
    # return args, data, train_dataloader, valid_dataloader, test_dataloader