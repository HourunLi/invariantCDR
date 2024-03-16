
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
from torch.nn import functional as F
from torch import nn
from tqdm import tqdm
from ..utils.utils import *
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
        self.writer = writer
        self.len = len(data["train"]["edge_list"]) # graph nums
        self.len_train = self.len - args.testlength - args.vallength
        self.len_val = args.vallength
        self.len_test = args.testlength
        self.nbsz = args.nbsz
        self.n_factors = args.n_factors
        self.delta_d = args.delta_d
        self.d = self.n_factors * self.delta_d
        self.interv_size_ratio = args.interv_size_ratio

        x = data["x"].to(args.device).clone().detach()
        self.x = [x for _ in range(self.len)] if len(x.shape) <= 2 else x
        self.edge_index_list_pre = [
            data["train"]["edge_index_list"][ix].long().to(args.device)
            for ix in range(self.len)
        ]
        neighbors_all = []
        for t in range(self.len):
            graph_data = Data(x=self.x[t], edge_index=self.edge_index_list_pre[t])
            graph = to_networkx(graph_data)
            sampler = NeibSampler(graph, self.nbsz)
            neighbors = sampler.sample().to(args.device)
            neighbors_all.append(neighbors)
        self.neighbors_all = torch.stack(neighbors_all).to(args.device)

        self.loss = EnvLoss(args)
        print("total length: {}, test length: {}".format(self.len, args.testlength))

    def train(self, epoch, data):
        args = self.args
        self.model.train()
        optimizer = self.optimizer
        
    def run(self):
        args = self.args
        min_epoch = args.min_epoch
        max_patience = args.patience
        patience = 0
        # 只优化名字中不含有ss的参数
        self.optimizer = optim.Adam(
            [parameters for name, parameters in self.model.named_parameters() if "ss" not in name],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        t_total0 = time.time()
        max_auc, max_test_auc, max_train_auc= 0, 0, 0
        with tqdm(range(1, args.max_epoch + 1)) as bar:
            for epoch in bar:
                t0 = time.time()
                epoch_losses, train_auc_list, val_auc_list, test_auc_list = self.train(
                    epoch, self.data["train"]
                )
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

                    metrics = "train_auc,val_auc,test_auc,epoch,test_train_auc,test_val_auc,test_test_auc".split(
                        ","
                    )
                    measure_dict = dict(
                        zip(
                            metrics,
                            [max_train_auc, max_auc, max_test_auc] + test_results,
                        )
                    )

                    patience = 0

                    filepath = "../checkpoint/" + self.args.dataset + ".pth"
                    torch.save(
                        {
                            "model_state_dict": self.model.state_dict(),
                            "cvae_state_dict": self.cvae.state_dict(),
                        },
                        filepath,
                    )
                # 如果验证集的accuracy没有提高，则增加耐心计数器。如果耐心计数器超过最大耐心值且训练轮数超过最小轮数，则提前停止训练。
                else:
                    patience += 1
                    if epoch > min_epoch and patience > max_patience:
                        break
                if epoch == 1 or epoch % self.args.log_interval == 0:
                    print(
                        "Epoch:{}, Loss: {:.4f}, Time: {:.3f}".format(
                            epoch, average_epoch_loss, time.time() - t0
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

        epoch_time = (time.time() - t_total0) / (epoch - 1)
        metrics = [max_train_auc, max_auc, max_test_auc] + test_results + [epoch_time]
        metrics_des = "train_auc,val_auc,test_auc,epoch,test_train_auc,test_val_auc,test_test_auc,epoch_time".split(
            ","
        )
        metrics_dict = dict(zip(metrics_des, metrics))
        df = pd.DataFrame([metrics], columns=metrics_des)
        print(df)
        return metrics_dict