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
import pandas as pd
import numpy as np

def drop_nodes(edge_index, node_num):
    print(f'Augmentation drop_nodes')
    # drop_user_num = int(user_max * 0.2)
    # drop_item_num = int(item_max * 0.2)
    # user_drop_idx = np.random.choice(user_max, drop_user_num, replace=False)
    # item_drop_idx = np.random.choice(item_max, drop_item_num, replace=False)
    
    # adjacency = torch.zeros((user_max, item_max))
    # adjacency[edge_index[0], edge_index[1]] = 1
    # adjacency[user_drop_idx, :] = 0
    # adjacency[:, item_drop_idx] = 0
    # edge_index_aug = torch.nonzero(adjacency).t()
    # return edge_index_aug
    drop_num = int(node_num * 0.15)
    idx_drop = np.random.choice(node_num, drop_num, replace=False)
    
    edge_index = edge_index.numpy()
    adj = torch.zeros((node_num, node_num))
    
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = torch.nonzero(adj).t()
    edge_index_aug = edge_index
    return edge_index_aug


def permute_edges(edge_index):
    print(f'Augmentation permute_edges')
    _, edge_num = edge_index.size()
    permute_num = int(edge_num * 0.2)
    edge_index = edge_index.transpose(0, 1).numpy()
    edge_index = edge_index[np.random.choice(edge_num, edge_num - permute_num, replace = False)]
    edge_index_aug = torch.tensor(edge_index).transpose(0, 1)
    return edge_index_aug

# construct user subgraph
def subUserGraph(edge_index, node_num, user_max):
    print(f'Augmentation subUserGraph')
    
    drop_num = int(user_max * 0.15)
    idx_drop = np.random.choice(user_max+1, drop_num, replace=False)
    
    edge_index = edge_index.numpy()
    adj = torch.zeros((node_num, node_num))
    
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    edge_index = torch.nonzero(adj).t()
    edge_index_aug = edge_index
    return edge_index_aug

    # sub_num = int(user_max * (1 - 0.2))
    # edge_index = edge_index.numpy()
    # user_sub = [np.random.randint(user_max, size=1)[0]]
    # item_neighbor = set([item for item in edge_index[1][edge_index[0] == user_sub[-1]]])
    # count = 1
    
    # while len(user_sub) <= sub_num:
    #     if count > user_max:
    #         break
    #     if len(item_neighbor) == 0:
    #         print("Warning: item_neighbor is zero when sampling in subUserGraph")
    #         break
    #     sample_item = np.random.choice(list(item_neighbor))
    #     user_neighbor = set([user for user in edge_index[0][edge_index[1] == sample_item]])
    #     sample_user = np.random.choice(list(user_neighbor))
    #     if sample_user in user_sub:
    #         continue
    #     user_sub.append(sample_user)
    #     count = count + 1
    #     print(f"subUserGraph count is {count}")
    #     item_neighbor.union(set([item for item in edge_index[1][edge_index[0] == user_sub[-1]]]))

    # user_drop = [n for n in range(user_max) if not n in user_sub]

    # edge_index = edge_index.numpy()

    # adjacency = torch.zeros((node_num, node_num))
    # adjacency[edge_index[0], edge_index[1]] = 1
    # adjacency[user_drop, :] = 0
    # edge_index_aug = torch.nonzero(adjacency).t()
    # return edge_index_aug


# construct item subgraph
def subItemGraph(edge_index, node_num, user_max, item_max):
    print(f'Augmentation subItemGraph')
    
    drop_num = int(item_max * 0.15)
    idx_drop = np.random.choice(item_max + 1, drop_num, replace=False)
    idx_drop = [item + user_max + 1 for item in idx_drop]
    
    edge_index = edge_index.numpy()
    adj = torch.zeros((node_num, node_num))
    
    adj[edge_index[0], edge_index[1]] = 1
    adj[:, idx_drop] = 0
    edge_index = torch.nonzero(adj).t()
    edge_index_aug = edge_index
    return edge_index_aug
    # sub_num = int(item_max * (1 - 0.2))
    # edge_index = edge_index.numpy()
    # item_sub = [np.random.randint(item_max, size=1)[0] + user_max + 1] 
    # user_neighbor = set([user for user in edge_index[0][edge_index[1] == item_sub[-1]]])
    # count = 0
    
    # while len(item_sub) <= sub_num:
    #     if count > item_max:
    #         break
    #     if len(user_neighbor) == 0:
    #         print("Warning: user_neighbor is zero when sampling in subItemGraph")
    #         break
    #     sample_user = np.random.choice(list(user_neighbor))
    #     item_neighbor = set([item for item in edge_index[1][edge_index[0] == sample_user]])
    #     sample_item = np.random.choice(list(item_neighbor))
    #     if sample_item in item_sub:
    #         continue
    #     item_sub.append(sample_item)
    #     count = count + 1
    #     print(f"subItemGraph count is {count}")
    #     user_neighbor.union(set([user for user in edge_index[0][edge_index[1] == item_sub[-1]]]))

    # item_drop = [user_max for n in range(item_max) if not n in item_sub]

    # edge_index = edge_index.numpy()

    # adjacency = torch.zeros((node_num, node_num))
    # adjacency[edge_index[0], edge_index[1]] = 1
    # adjacency[:, item_drop] = 0
    # edge_index_aug = torch.nonzero(adjacency).t()
    # return edge_index_aug

def augmentData(args, data, data_aug):
    data_aug["train"]["edge_lists"] = []
    for idx, edge_index in enumerate(data["train"]["edge_lists"]):
        if args.aug == 'dnodes':
            edge_index_aug = drop_nodes(deepcopy(edge_index), data["node_num"][idx])
        elif args.aug == 'pedges':
            edge_index_aug = permute_edges(deepcopy(edge_index))
        elif args.aug == 'subUserGraph':
            edge_index_aug = subUserGraph(deepcopy(edge_index), data["user_max"][idx])
        elif args.aug == 'subItemGraph':
            edge_index_aug = subItemGraph(deepcopy(edge_index), data["user_max"][idx], data["item_max"][idx])
        elif args.aug == 'none':
            edge_index_aug = deepcopy(edge_index)
            # edge_index_aug.x = torch.ones((data.edge_index.max() + 1, 1))
        elif args.aug == 'random4':
            n = np.random.randint(4)
            if n == 0:
                edge_index_aug = drop_nodes(deepcopy(edge_index), data["node_num"][idx])
            elif n == 1:
                edge_index_aug = permute_edges(deepcopy(edge_index))
            elif n == 2:
                edge_index_aug = subUserGraph(deepcopy(edge_index), data["node_num"][idx], data["user_max"][idx])
            elif n == 3:
                edge_index_aug = subItemGraph(deepcopy(edge_index), data["node_num"][idx], data["user_max"][idx], data["item_max"][idx])
            else:
                print('sample error')
                assert False
        else:
            print('augmentation error')
            assert False
        data_aug["train"]["edge_lists"].append(edge_index_aug)
    return 

# only for dual user inter now, need for branch if add new scnerios
def loadTrainFile(args, data):
    # x_list is for node features, presented by [node_num, node_feature]
    # edge_list is for edge, presented by [edge_num, 2]
    x_list, edge_list = [], []
    for _, cur_domain in enumerate(args.domains):
        cur_src_data_dir = os.path.join("./datasets/"+str(args.task) + "/dataset/", cur_domain + "/train.txt")
        print(f'Loading {cur_domain}: {cur_src_data_dir}')
        edges = []
        max_user, max_item = 0, 0
        with codecs.open(cur_src_data_dir, "r", encoding="utf-8") as infile:
            for line in infile:
                line = line.strip().split("\t")
                user, item= int(line[0]), int(line[1])
                max_user, max_item= max(max_user, user), max(max_item, item)
                edges.append([user, item])
        data["user_max"].append(max_user)
        data["item_max"].append(max_item)
        data["node_num"].append(max_user + max_item + 2)
        print(f"cur_domain's user_max:{data['user_max']}, item_max:{data['item_max']}, node_num:{data['node_num']}")
        """
        user and item both start at 0. Put item index after user, to unify the graph
        dual user inter数据集中user是共享的，只有大图需要处理。
        """
        edges = [[user, item + max_user] for user, item in edges]
        edges = torch.tensor(edges).transpose(0, 1)
        x = torch.ones((data["node_num"][-1], 1))
        args.node_feature = 1
        edge_list.append(edges)
        x_list.append(x)

    data["x"] = x_list
    data["train"]["edge_lists"] = edge_list

    return 

def load_data(args):
    if "dual" in args.task:
        filename = args.domains.split("_")
        args.domains = []
        args.domains.append(filename[0] + "_" + filename[1])
        args.domains.append(filename[1] + "_" + filename[0])
    else:
        args.domains = args.domains.split('_')
    print("Loading domains:", args.domains)

    """
    data = {"x":[num_domains, node_num, 1]], "train" : {"edge_lists":[edge_num, 2]}, 
        "user_max": [domain_num, 1], "item_max": [domain_num, 1], "node_num": [domain_num, 1], 
        "domains": [domain_num, 1], "domain_num": int}
    """
    data = {}
    data["user_max"], data["item_max"], data["node_num"] = [], [], []
    data["train"], data["valid"], data["test"] = {}, {}, {}
    data["domains"] = [domain for domain in args.domains]
    data["domain_num"], args.num_domains = len(args.domains), len(args.domains)
    
    loadTrainFile(args, data)
    
    data_aug = {}
    data_aug["train"] = {}
    data_aug["x"], data_aug["domains"], data_aug["domain_num"] = deepcopy(data["x"]), deepcopy(data["domains"]), data["domain_num"]
    augmentData(args, data, data_aug)
    return args, data, data_aug

