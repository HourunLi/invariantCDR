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

class invariantCDR(nn.Module):
    def __init__(self, args):
        super(invariantCDR, self).__init__()
        self.model = DGCL(args)
        self.args = args
        self.user_embedding_list = []
        self.item_embedding_lsit = []
        
        for i in range(self.args.num_domains):
            self.user_embedding_list.append(nn.Embedding(args.user_max[i], args.latent_dim))
            self.item_embedding_lsit.append(nn.Embedding(args.item_max[i] + 1, args.latent_dim, padding_idx=0))
        
        self.user_embedding_list = nn.ModuleList(self.user_embedding_list)
        self.item_embedding_lsit = nn.ModuleList(self.item_embedding_lsit)
        
class DGCL(nn.Module):
    def __init__(self, args):
        super(DGCL, self).__init__()
        # print(args)
        # Namespace(DS='MUTAG', JK='sum', aug='random4', batch=128, debug=False, drop_ratio=0.3, 
        # epoch=30, fe=0, head_layers=1, latent_dim=126, log_dir='log', log_interval=5, lr=0.0001, 
        # num_gc_layers=4, num_latent_factors=3, num_workers=8, pool='mean', proj=1, residual=0, seed=32, tau=0.2)
        self.args = args
        self.device = args.device
        self.encoder = DisenEncoder(args)
        self.T = args.tau # The temperature parameter for contrastive learning, set to 0.2
        self.K = args.num_latent_factors # 3
        self.latent_dim = args.latent_dim
        self.d = self.latent_dim // self.K
        self.node_dim = args.node_dim
        self.num_layers = args.num_layers
        self.center_v = torch.rand((self.K, self.d), requires_grad=True).to(self.device)
        self.init_emb()

    # the init_emb method is responsible for initializing the weights and biases of all linear layers in the neural network
    def init_emb(self):
        initrange = -1.5 / self.latent_dim
        for m in self.modules():
            if isinstance(m, nn.Linear): # For each module, it checks if the module is an instance of a linear layer (nn.Linear).
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, edge_index):
        graph_dis_emb, nod_dis_emb = self.encoder(edge_index)
        return graph_dis_emb, nod_dis_emb

    def loss_cal(self, x, x_aug):
        T = self.T #  temperature parameter for scaling the similarity scores
        T_c = 0.2 #  temperature parameter used for the softmax function in the calculation of cluster probabilities.
        B, H, d = x.size() # batch, k factor, dimension
        ck = F.normalize(self.center_v)
        # compute the similarity scores between the normalized embeddings x and the normalized cluster centers ck
        # 说白了就是dot product来计算相似度
        p_k_x_ = torch.einsum('bkd,kd->bk', F.normalize(x, dim=-1), ck)
        #  increasing the contrast between the highest and lower values.
        # 说白了就是使得区分度更大, adjust sensitivity
        p_k_x = F.softmax(p_k_x_ / T_c, dim=-1)
        # 计算dimension维度的二范数
        # x: torch.Size([128, 3, 42])
        # x_abs: torch.Size([128, 3])
        x_abs = x.norm(dim=-1)
        # print(x.size())
        # print(x_abs.size())
        x_aug_abs = x_aug.norm(dim=-1)
        x = torch.reshape(x, (B * H, d))
        x_aug = torch.reshape(x_aug, (B * H, d))
        # torch.reshape(x_abs, (B * H, 1)）:torch.Size([384, 1])
        # 把最后那个1维度去掉
        # torch.Size([384])
        x_abs = torch.squeeze(torch.reshape(x_abs, (B * H, 1)), 1)
        x_aug_abs = torch.squeeze(torch.reshape(x_aug_abs, (B * H, 1)), 1)
        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / (1e-8 + torch.einsum('i,j->ij', x_abs, x_aug_abs))
        sim_matrix = torch.exp(sim_matrix / T)
        # The diagonal elements of the similarity matrix represent the positive pairs
        # (original and augmented embeddings of the same graph).
        pos_sim = sim_matrix[range(B * H), range(B * H)]
        # The contrastive loss is computed as the log ratio of the positive similarity scores 
        # to the sum of all similarity scores, excluding the positive pairs.
        score = pos_sim / (sim_matrix.sum(dim=-1) - pos_sim)
        p_y_xk = score.view(B, H)
        # Overall, this loss function encourages the model to learn embeddings that are similar for original 
        # and augmented versions of the same graph while also being discriminative with respect to different cluster centers. 
        q_k = torch.einsum('bk,bk->bk', p_k_x, p_y_xk)
        q_k = F.normalize(q_k, dim=-1)
        elbo = q_k * (torch.log(p_k_x) + torch.log(p_y_xk) - torch.log(q_k))
        loss = - elbo.view(-1).mean()
        return loss


class DisenEncoder(torch.nn.Module):
    def __init__(self, args):
        super(DisenEncoder, self).__init__()
        # print(args)
        # Namespace(DS='MUTAG', JK='sum', aug='random4', batch=128, debug=False, drop_ratio=0.3, 
        # epoch=30, fe=0, head_layers=1, latent_dim=126, log_dir='log', log_interval=5, lr=0.0001, 
        # num_gc_layers=4, num_latent_factors=3, num_workers=8, pool='mean', proj=1, residual=0, seed=32, tau=0.2)
        self.args = args
        self.device = args.device
        self.num_features = args.num_features
        self.K = args.K # k latent factor (hyper parameter)
        self.d = args.latent_dim // self.K # dimension for each latent factor
        self.num_layers = args.num_layers 
        self.head_layers = args.head_layers # The number of head layers in the encoder, used for further processing or transformation of the embeddings after the graph convolutional layers.
        self.gc_layers = self.num_layers - self.head_layers
        self.if_proj_head = args.if_proj_head # A projection head is an additional neural network layer (or layers) applied to the embeddings, often used in contrastive learning to improve representation learning.
        self.drop_ratio = args.drop_ratio
        self.graph_pooling = args.graph_pooling
        if self.graph_pooling == "sum" or self.graph_pooling == 'add':
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")
        self.JK = args.JK
        if self.JK == 'last':
            pass
        elif self.JK == 'sum':
            self.JK_proj = Linear(self.gc_layers * args.latent_dim, args.latent_dim)
        else:
            assert False
        self.residual = args.residual
        
        # build the graph convolutional network 
        self.convs = torch.nn.ModuleList() # convolutional layers
        self.bns = torch.nn.ModuleList() # batch normalization layers 
        # build the disentangled representation learning mechanism.
        self.disen_convs = torch.nn.ModuleList() # disentangled convolutional layers 
        self.disen_bns = torch.nn.ModuleList() # disentangled batch normalization layers

        for i in range(self.gc_layers):
            if i == 0:
                nn = Sequential(Linear(args.node_dim, args.latent_dim), ReLU(), Linear(args.latent_dim, args.latent_dim))
            else:
                nn = Sequential(Linear(args.latent_dim, args.latent_dim), ReLU(), Linear(args.latent_dim, args.latent_dim))
            conv = GINConv(nn) # Graph Isomorphism Network, GIN
            bn = torch.nn.BatchNorm1d(args.latent_dim)

            self.convs.append(conv)
            # A batch normalization layer (bn) is created for the embedding dimension (emb_dim)
            self.bns.append(bn)

        for i in range(self.K): # 外循环是K factor，内循环是head layers，说白了就是有几个encoder
            for j in range(self.head_layers):
                if j == 0:
                    nn = Sequential(Linear(args.latent_dim, self.d), ReLU(), Linear(self.d, self.d))
                else:
                    nn = Sequential(Linear(self.d, self.d), ReLU(), Linear(self.d, self.d))
                conv = GINConv(nn)
                bn = torch.nn.BatchNorm1d(self.d)
                self.disen_convs.append(conv)
                self.disen_bns.append(bn)
        # projection head就是k factor对应的变换层
        self.proj_heads = torch.nn.ModuleList()
        for i in range(self.K):
            nn = Sequential(Linear(self.d, self.d), ReLU(inplace=True), Linear(self.d, self.d))
            self.proj_heads.append(nn)

    def forward(self, x, edge_index):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(self.device)
        h_node = self._normal_conv(x, edge_index)
        h_graph_multi, h_node_multi = self._disen_conv(h_node, edge_index)
        return h_graph_multi, h_node_multi
    
    def get_embeddings(self, data):
        ret = []
        y = []
        with torch.no_grad():
            data.to(device = self.device)
            x, edge_index, batch = data.x, data.edge_index, data.batch
            if x is None:
                x = torch.ones((batch.shape[0], 1)).to(self.device)
            x, _ = self.forward(x, edge_index, batch)
            B, K, d = x.size()
            x = x.view(B, K * d)
            ret.append(x.cpu().numpy())
            y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y