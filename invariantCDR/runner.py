
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
from invariantCDR.model import *

class Runner(object):
    def __init__(self, args, model, data, data_aug, writer=None, **kwargs):
        seed_everything(args.seed)
        self.args = args
        self.model = model
        self.data = data
        self.data_aug = data_aug
        # self.DGCL = DGCL(args)
        self.device = args.device
        self.length = len(data["train"]["edge_list"]) # number of graphs

    def train(self, epoch, data, data_aug):
        # ret: epoch_losses, train_auc_list, val_auc_list, test_auc_list 
        args = self.args
        self.model.train()
        optimizer = self.optimizer
        # for graph in data, data_aug
        for idx in range(self.length):
            # graph = data["edge_list"][idx] # get the idx grpah
            # graph_aug = data_aug["edge_list"][idx]
            optimizer.zero_grad() # 重置梯度
            node_num, _ = data.x.size()
            data = data.to(self.device)
            # 得到图和图节点的disentangled representation
            # 使用的是K个相同的DGCL encoder（里面包含contrastive learning）
            graph_emb, node_emb = self.model.DGCL(data, idx)
            # 得到data_aug的edge_idx
            edge_idx = data_aug["edge_list"][idx].numpy()
            _, edge_num = edge_idx.shape
            # 在边中的实体的id
            idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]
            node_num_aug = len(idx_not_missing)
            # 从data_aug.x和data_aug.batch筛选出有边连接的点
            # 之前的data_aug.x应该是节点全包含的，所以要筛选出部分便于后续训练
            data_aug.x = data_aug.x[idx_not_missing] #[xx,7]
            # data_aug.batch = data.batch[idx_not_missing]
            idx_dict = {idx_not_missing[n]: n for n in range(node_num_aug)}
            # 去掉自旋边
            edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if
                        not edge_idx[0, n] == edge_idx[1, n]]
            # transpose [2, xx] to [xx, 2]
            data_aug["edge_list"][idx]= torch.tensor(edge_idx).transpose_(0, 1)
            data_aug = data_aug.to(self.device)
            graph_emb_aug, node_emb_aug = self.model.DGCL(data_aug, idx)
            # print(x.size(), x_aug.size())
            # torch.Size([128, 3, 42])  torch.Size([128, 3, 42] [batchsize, k factor, dim]
            loss = self.model.loss_cal(node_emb, node_emb_aug) # contrastive loss,计算原数据和增强后数据的嵌入向量相似度
            loss_all += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
        
        
    def run(self):
        args = self.args
        min_epoch = args.min_epoch
        max_patience = args.patience
        patience = 0
        optimizer = args.optimizer
        self.model.train()
        # 对每个图进行DGCL，得到disentangle learning
        emb, y = self.model.encoder.get_embeddings(
            [self.data["train"]["edge_lists"][ind].long().to(args.device) for ind in range(self.len)]
        )
        max_auc, max_test_auc, max_train_auc = 0, 0, 0
        train_start = time.time()
        with tqdm(range(1, args.epoch+1)) as bar:
            for epoch in bar:
                epoch_start = time.time()
                epoch_losses, train_auc_list, val_auc_list, test_auc_list = self.train(epoch, self.data, self.data_aug)
                average_epoch_loss = epoch_losses
                average_train_auc = np.mean(train_auc_list)
                average_val_auc = np.mean(val_auc_list)
                average_test_auc = np.mean(test_auc_list)

                # 如果验证集的accuracy超过当前最大的accuracy，那么更新accuracy，并且在测评集上评估模型
                if average_val_auc > max_auc:
                    max_auc = average_val_auc
                    max_test_auc = average_test_auc
                    max_train_auc = average_train_auc
                    test_results = self.test(epoch, self.data["test"])
                    metrics = "train_auc,val_auc,test_auc,epoch,test_train_auc,test_val_auc,test_test_auc".split(",")
                    measure_dict = dict(
                        zip(
                            metrics,
                            [max_train_auc, max_auc, max_test_auc] + test_results,
                        )
                    )
                    patience = 0
                    filepath = "./checkpoint/" + self.args.dataset + ".pth"
                    torch.save({"model_state_dict": self.model.state_dict()}, filepath)
                else:
                    patience += 1
                    if epoch > min_epoch and patience > max_patience:
                        break
                if epoch == 1 or epoch % self.args.log_interval == 0:
                    print(
                        "Epoch:{}, Loss: {:.4f}, Time: {:.3f}".format(
                            epoch, average_epoch_loss, time.time() - epoch_start
                        )
                    )
                    print(
                        f"Current: Epoch:{epoch}, Train AUC:{average_train_auc:.4f}, Val AUC: {average_val_auc:.4f}, Test AUC: {average_test_auc:.4f}"
                    )

                    print(
                        f"Train: Epoch:{test_results[0]}, Train AUC:{max_train_auc:.4f}, Val AUC: {max_auc:.4f}, Test AUC: {max_test_auc:.4f}"
                    )
                    print(
                        f"Test: Epoch:{test_results[0]}, Train AUC:{test_results[1]:.4f}, Val AUC: {test_results[2]:.4f}, Test AUC: {test_results[3]:.4f}"
                    )
        avg_epoch_time = (time.time() - train_start) / (epoch - 1)
        metrics = [max_train_auc, max_auc, max_test_auc] + test_results + [avg_epoch_time]
        metrics_des = "train_auc,val_auc,test_auc,epoch,test_train_auc,test_val_auc,test_test_auc,epoch_time".split(",")
        metrics_dict = dict(zip(metrics_des, metrics))
        df = pd.DataFrame([metrics], columns=metrics_des)
        print(df)
        return metrics_dict