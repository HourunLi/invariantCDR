
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
    def __init__(self, args, model, data, writer=None, **kwargs):
        seed_everything(args.seed)
        self.args = args
        self.model = model
        self.data = data
        self.DGCL = DGCL(args)
        self.len = len(data["train"]["edge_list"]) # number of graphs

    def train(self, epoch, data):
        # epoch_losses, train_auc_list, val_auc_list, test_auc_list 
        args = self.args
        self.model.train()
        optimizer = self.optimizer
        # embeddings 
        
        
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
                epoch_losses, train_auc_list, val_auc_list, test_auc_list = self.train(epoch, self.data["train"])
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