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

class DGCL(nn.Module):
    def __init__(self, num_features, hidden_dim, num_layer, device, args):
        super(DGCL, self).__init__()
        # print(args)
        # Namespace(DS='MUTAG', JK='sum', aug='random4', batch=128, debug=False, drop_ratio=0.3, 
        # epoch=30, fe=0, head_layers=1, hidden_dim=126, log_dir='log', log_interval=5, lr=0.0001, 
        # num_gc_layers=4, num_latent_factors=3, num_workers=8, pool='mean', proj=1, residual=0, seed=32, tau=0.2)
        self.args = args
        self.device = device
        self.T = self.args.tau # The temperature parameter for contrastive learning, set to 0.2
        self.K = args.num_latent_factors # 3
        self.embedding_dim = hidden_dim
        self.d = self.embedding_dim // self.K

        self.center_v = torch.rand((self.K, self.d), requires_grad=True).to(device)

        self.encoder = DisenEncoder(
            num_features=num_features,
            emb_dim=hidden_dim,
            num_layer=num_layer,
            K=args.num_latent_factors,
            head_layers=args.head_layers,
            device=device,
            args=args,
            if_proj_head=args.proj > 0, # A flag or parameter related to projection, set to 1.
            drop_ratio=args.drop_ratio, 
            graph_pooling=args.pool,
            JK=args.JK,
            residual=args.residual > 0
        )

        self.init_emb()
    # the init_emb method is responsible for initializing the weights and biases of all linear layers in the neural network
    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear): # For each module, it checks if the module is an instance of a linear layer (nn.Linear).
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch, num_graphs):
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)
        z_graph, _ = self.encoder(x, edge_index, batch)
        return z_graph

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
    def __init__(self, num_features, emb_dim, num_layer, K, head_layers, if_proj_head=False, drop_ratio=0.0,
                 graph_pooling='add', JK='last', residual=False, device=None, args=None):
        super(DisenEncoder, self).__init__()
        # print(args)
        # Namespace(DS='MUTAG', JK='sum', aug='random4', batch=128, debug=False, drop_ratio=0.3, 
        # epoch=30, fe=0, head_layers=1, hidden_dim=126, log_dir='log', log_interval=5, lr=0.0001, 
        # num_gc_layers=4, num_latent_factors=3, num_workers=8, pool='mean', proj=1, residual=0, seed=32, tau=0.2)
        self.args = args
        self.device = device
        self.num_features = num_features
        self.K = K
        self.d = emb_dim // self.K
        self.num_layer = num_layer
        self.head_layers = head_layers # The number of head layers in the encoder, used for further processing or transformation of the embeddings after the graph convolutional layers.
        self.gc_layers = self.num_layer - self.head_layers
        self.if_proj_head = if_proj_head # A projection head is an additional neural network layer (or layers) applied to the embeddings, often used in contrastive learning to improve representation learning.
        self.drop_ratio = drop_ratio
        self.graph_pooling = graph_pooling
        if self.graph_pooling == "sum" or self.graph_pooling == 'add':
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")
        self.JK = JK
        if JK == 'last':
            pass
        elif JK == 'sum':
            self.JK_proj = Linear(self.gc_layers * emb_dim, emb_dim)
        else:
            assert False
        self.residual = residual
        # build the graph convolutional network 
        self.convs = torch.nn.ModuleList() # convolutional layers
        self.bns = torch.nn.ModuleList() # batch normalization layers 
        # build the disentangled representation learning mechanism.
        self.disen_convs = torch.nn.ModuleList() # disentangled convolutional layers 
        self.disen_bns = torch.nn.ModuleList() # disentangled batch normalization layers

        for i in range(self.gc_layers):
            if i == 0:
                nn = Sequential(Linear(num_features, emb_dim), ReLU(), Linear(emb_dim, emb_dim))
            else:
                nn = Sequential(Linear(emb_dim, emb_dim), ReLU(), Linear(emb_dim, emb_dim))
            conv = GINConv(nn) # Graph Isomorphism Network, GIN
            bn = torch.nn.BatchNorm1d(emb_dim)

            self.convs.append(conv)
            # A batch normalization layer (bn) is created for the embedding dimension (emb_dim)
            self.bns.append(bn)

        for i in range(self.K): # 外循环是K factor，内循环是head layers，说白了就是有几个encoder
            for j in range(self.head_layers):
                if j == 0:
                    nn = Sequential(Linear(emb_dim, self.d), ReLU(), Linear(self.d, self.d))
                else:
                    nn = Sequential(Linear(self.d, self.d), ReLU(), Linear(self.d, self.d))
                conv = GINConv(nn)
                # 进行归一化
                bn = torch.nn.BatchNorm1d(self.d)

                self.disen_convs.append(conv)
                self.disen_bns.append(bn)
        # projection head就是k factor对应的变换层
        self.proj_heads = torch.nn.ModuleList()
        for i in range(self.K):
            nn = Sequential(Linear(self.d, self.d), ReLU(inplace=True), Linear(self.d, self.d))
            self.proj_heads.append(nn)
    # This method applies standard graph convolutional layers to the input graph features x
    def _normal_conv(self, x, edge_index, batch):
        xs = []
        for i in range(self.gc_layers): # For each layer, it applies a graph convolution (self.convs[i]) followed by batch normalization (self.bns[i]).
            # x是特征值，edge_index是边
            # 先经历一次卷机进行message passing，然后过一次batch normalization
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            # 如果是最后一层，那么直接输出，否则过一次激活函数。
            if i == self.gc_layers - 1:
                x = F.dropout(x, self.drop_ratio, training=self.training)
            else:
                x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
            # 根据residual设置xs，说白了就是为后面jumping knwoledge 做准备，需要获取之前layer的信息知识
            if self.residual and i > 0: # If residual connections are enabled (self.residual), the output of the current layer is added to the output of the previous layer.
                x += xs[i - 1]
            xs.append(x)
        if self.JK == 'last':
            return xs[-1]
        elif self.JK == 'sum':
            return self.JK_proj(torch.cat(xs, dim=-1))

    def _disen_conv(self, x, edge_index, batch):
        x_proj_list = []
        x_proj_pool_list = []
        for i in range(self.K):
            x_proj = x
            for j in range(self.head_layers):
                # 因为所有k个factor对应的conv都放一个modulelist了，所以先计算index
                tmp_index = i * self.head_layers + j
                # 跟前面的处理方式是一样的
                x_proj = self.disen_convs[tmp_index](x_proj, edge_index)
                x_proj = self.disen_bns[tmp_index](x_proj)
                # 这里没有dropout
                if j != self.head_layers - 1:
                    x_proj = F.relu(x_proj)
            x_proj_list.append(x_proj)
            # x_proj_pool_list是想要获得一个graph level的representation (based on the disentangled node representations.)
            x_proj_pool_list.append(self.pool(x_proj, batch))
        # print(f"the length of x_proj_pool_list is {len(x_proj_pool_list)} and {x_proj_pool_list[0].size()}")
        # the length of x_proj_pool_list is 3 and torch.Size([60, 42])
        if self.if_proj_head:
            x_proj_pool_list = self._proj_head(x_proj_pool_list)
        # dim = 0是默认参数， 把x_proj_pool_list把第一位堆叠起来，其实等价于直接转换为tensor
        x_graph_multi = torch.stack(x_proj_pool_list)
        # print(f"the size of x_graph_multi is {x_graph_multi.size()}") 
        # the size of x_graph_multi is torch.Size([3, 128, 42])
        x_node_multi = torch.stack(x_proj_list)
        # print(f"the size of x_node_multi is {x_node_multi.size()}") 
        # the size of x_node_multi is torch.Size([3, 2298, 42])
        # contiguous有利于后续连续访问性能
        x_graph_multi = x_graph_multi.permute(1, 0, 2).contiguous()
        x_node_multi = x_node_multi.permute(1, 0, 2).contiguous()
        return x_graph_multi, x_node_multi

    def _proj_head(self, x_proj_pool_list):
        ret = []
        for k in range(self.K):
            x_graph_proj = self.proj_heads[k](x_proj_pool_list[k])
            ret.append(x_graph_proj)
        return ret

    def forward(self, x, edge_index, batch):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)
        h_node = self._normal_conv(x, edge_index, batch)
        h_graph_multi, h_node_multi = self._disen_conv(h_node, edge_index, batch)
        return h_graph_multi, h_node_multi

    def get_embeddings(self, loader):
        device = self.device
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data = data[0]
                # print(len(data), data)
                # for i in range(len(data)):
                #     print(data[i])
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(device)
                x, _ = self.forward(x, edge_index, batch)
                B, K, d = x.size()
                x = x.view(B, K * d)
                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y

