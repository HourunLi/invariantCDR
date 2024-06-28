import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal
from torch.nn.modules.module import Module
from torch.nn.modules.module import Module
from scipy.sparse import csr_matrix
from time import time


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, adj):
        x = self.leakyrelu(self.gc1(x, adj))
        return x

class VGAE(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha):
        super(GCN, self).__init__()
        self.gc_mean = GraphConvolution(nfeat, nhid)
        self.gc_logstd = GraphConvolution(nfeat, nhid)
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.nhid = nhid

    def _kld_gauss(self, mu_1, logsigma_1, mu_2, logsigma_2):
        """Using std to compute KLD"""
        # sigma_1 = torch.exp(0.1 + 0.9 * F.softplus(sigma_1))
        # sigma_2 = torch.exp(0.1 + 0.9 * F.softplus(sigma_2))
        sigma_1 = 0.1 + 0.9 * F.softplus(torch.exp(logsigma_1))
        sigma_2 = 0.1 + 0.9 * F.softplus(torch.exp(logsigma_2))
        q_target = Normal(mu_1, sigma_1)
        q_context = Normal(mu_2, sigma_2)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return kl

    def encode(self, x, adj):
        mean = self.gc_mean(x, adj)
        logstd = self.gc_logstd(x, adj)
        gaussian_noise = torch.randn(x.size(0), self.nhid)
        if self.gc_mean.training:
            sampled_z = gaussian_noise * torch.exp(logstd) + mean
            self.kld_loss = self._kld_gauss(mean, logstd, torch.zeros_like(mean), torch.ones_like(logstd))
        else :
            sampled_z = mean
        return sampled_z

    def forward(self, x, adj):
        x = self.encode(x, adj)
        return x

# def dot_product_decode(Z):
# 	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
# 	return A_pred

class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        # self.weight = self.glorot_init(in_features, out_features)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def glorot_init(self, input_dim, output_dim):
        init_range = np.sqrt(6.0 / (input_dim + output_dim))
        initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
        return nn.Parameter(initial / 2)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError

class LightGCN(BasicModel):
    def __init__(self, args, n_users, m_items, adj):
        super(LightGCN, self).__init__()
        self.args = args
        self.num_users  = n_users
        self.num_items  = m_items
        self.Graph = adj
        self.__init_weight()

    def __init_weight(self):
        self.latent_dim = self.args.latent_dim
        self.n_layers = self.args.num_layers
        self.keep_prob = self.args.keep_prob
        self.A_split = self.args.A_split
        self.f = nn.Sigmoid()
        print(f"lgn is already to go(dropout:{self.args.dropout})")
        
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def computer(self, users_emb, items_emb):
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.args.dropout:
            if self.training:
                # print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users_emb_agg, items_emb_agg = torch.split(light_out, [self.num_users, self.num_items])
        return users_emb_agg, items_emb_agg

    def forward(self, users_emb_specific, users_emb_shared, items_emb):
        users_emb_specific_agg, items_emb_specific_agg = self.computer(users_emb_specific, items_emb)
        users_emb_shared_agg, items_emb_shared_agg = self.computer(users_emb_shared, items_emb)
        items_emb_agg = (items_emb_specific_agg + items_emb_shared_agg) / 2

        return users_emb_specific_agg, users_emb_shared_agg, items_emb_agg

