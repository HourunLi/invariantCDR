
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
from invariantCDR.model import DGCL, invariantCDR, DisenEncoder

class Runner(object):
    def __init__(self, args, model, data, data_aug, writer=None, **kwargs):
        seed_everything(args.seed)
        self.args = args
        self.model = model
        self.data, self.data_aug = data, data_aug
        self.lr = args.lr
        self.length = data["domain_num"]# number of graphs
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = args.lr)
        self.max_patience = args.patience

    def train(self, data, data_aug):
        loss_all = 0
        self.model.train()
        optimizer = self.optimizer
        optimizer.zero_grad()
        for idx in range(self.length):
            node_num, _ = data["x"][idx].size()
            # move2GPU(data, device)
            graph_emb, node_emb = self.model.DGCL(data, idx)
            
            edge_idx_aug = data_aug["train"]["edge_lists"][idx].numpy()
            _, edge_num = edge_idx_aug.shape
            idx_not_missing = [n for n in range(node_num) if (n in edge_idx_aug[0] or n in edge_idx_aug[1])]
            node_num_aug = len(idx_not_missing)
            data_aug["x"][idx] = data_aug["x"][idx][idx_not_missing] #[xx,7]
            idx_dict = {idx_not_missing[n]: n for n in range(node_num_aug)}
            edge_idx_aug = [[idx_dict[edge_idx_aug[0, n]], idx_dict[edge_idx_aug[1, n]]] for n in range(edge_num) if
                        not edge_idx_aug[0, n] == edge_idx_aug[1, n]]
            data_aug["train"]["edge_lists"][idx]= torch.tensor(edge_idx_aug).transpose_(0, 1)
            # move2GPU(data_aug, device)
            graph_emb_aug, node_emb_aug = self.model.DGCL(data_aug, idx)

            print("----------------------------------------------x, x_aug--------------------------------------------")
            print(graph_emb, graph_emb_aug)
        
            loss = self.model.DGCL.loss_cal(graph_emb, graph_emb_aug) # contrastive loss,计算原数据和增强后数据的嵌入向量相似度
            loss_all += loss.item()
            loss.backward()
            optimizer.step()
            
        avg_epoch_loss = loss_all / self.length
        return avg_epoch_loss
        # ret: epoch_losses, train_auc_list, val_auc_list, test_auc_list 
        
    def run(self):
        args = self.args
        min_epoch = args.min_epoch
        patience, max_patience = 0, self.max_patience
        # 对每个图进行DGCL，得到disentangle learning
        # emb, y = self.model.encoder.get_embeddings(
        #     [self.data["train"]["edge_lists"][ind].long().to(args.device) for ind in range(self.len)]
        # )
        max_auc, max_test_auc, max_train_auc = 0, 0, 0
        train_start = time.time()
        with tqdm(range(1, args.epoch+1)) as bar:
            for epoch in bar:
                epoch_start = time.time()
                avg_epoch_loss, train_auc_list, val_auc_list, test_auc_list = self.train(self.data, self.data_aug)
                print('loss %.4f' % avg_epoch_loss)
        #         avg_epoch_loss = epoch_losses
        #         avg_train_auc = np.mean(train_auc_list)
        #         avg_val_auc = np.mean(val_auc_list)
        #         avg_test_auc = np.mean(test_auc_list)

        #         # 如果验证集的accuracy超过当前最大的accuracy，那么更新accuracy，并且在测评集上评估模型
        #         if avg_val_auc > max_auc:
        #             max_auc = avg_val_auc
        #             max_test_auc = avg_test_auc
        #             max_train_auc = avg_train_auc
        #             test_results = self.test(epoch, self.data["test"])
        #             metrics = "train_auc,val_auc,test_auc,epoch,test_train_auc,test_val_auc,test_test_auc".split(",")
        #             measure_dict = dict(
        #                 zip(
        #                     metrics,
        #                     [max_train_auc, max_auc, max_test_auc] + test_results,
        #                 )
        #             )
        #             patience = 0
        #             filepath = "./checkpoint/" + self.args.dataset + ".pth"
        #             torch.save({"model_state_dict": self.model.state_dict()}, filepath)
        #         else:
        #             patience += 1
        #             if epoch > min_epoch and patience > max_patience:
        #                 break
        #         if epoch == 1 or epoch % self.args.log_interval == 0:
        #             print(
        #                 "Epoch:{}, Loss: {:.4f}, Time: {:.3f}".format(
        #                     epoch, avg_epoch_loss, time.time() - epoch_start
        #                 )
        #             )
        #             print(
        #                 f"Current: Epoch:{epoch}, Train AUC:{avg_train_auc:.4f}, Val AUC: {avg_val_auc:.4f}, Test AUC: {avg_test_auc:.4f}"
        #             )

        #             print(
        #                 f"Train: Epoch:{test_results[0]}, Train AUC:{max_train_auc:.4f}, Val AUC: {max_auc:.4f}, Test AUC: {max_test_auc:.4f}"
        #             )
        #             print(
        #                 f"Test: Epoch:{test_results[0]}, Train AUC:{test_results[1]:.4f}, Val AUC: {test_results[2]:.4f}, Test AUC: {test_results[3]:.4f}"
        #             )
        # avg_epoch_time = (time.time() - train_start) / (epoch - 1)
        # metrics = [max_train_auc, max_auc, max_test_auc] + test_results + [avg_epoch_time]
        # metrics_des = "train_auc,val_auc,test_auc,epoch,test_train_auc,test_val_auc,test_test_auc,epoch_time".split(",")
        # metrics_dict = dict(zip(metrics_des, metrics))
        # df = pd.DataFrame([metrics], columns=metrics_des)
        # print(df)
        # return metrics_dict
        return