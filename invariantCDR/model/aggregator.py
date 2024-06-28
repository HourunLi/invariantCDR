import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
import numpy as np
import math
from copy import deepcopy
from itertools import repeat
from torch_geometric.data import DataLoader, InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data
from torch_geometric.utils import degree
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, global_max_pool

class BehaviorAggregator(nn.Module):
    def __init__(self, args):
        super(BehaviorAggregator, self).__init__()
        self.args = args
        self.aggregator = self.args.aggregator
        self.lambda_a = self.args.lambda_a
        embedding_dim = self.args.latent_dim
        dropout_rate = self.args.dropout

        self.W_agg = nn.Linear(embedding_dim, embedding_dim, bias=False)
        if self.aggregator in ["user_attention"]:
            self.W_att = nn.Sequential(nn.Linear(embedding_dim, embedding_dim),
                                     nn.Tanh())
            self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
    
    def forward(self, id_emb, sequence_emb, score):
        out = id_emb
        if self.aggregator == "mean":
            out = self.mean_pooling(sequence_emb)
        elif self.aggregator == "user_attention":
            out = self.user_attention_pooling(id_emb, sequence_emb)
        elif self.aggregator == "item_similarity":
            out = self.item_similarity_pooling(sequence_emb, score)
        else:
            print("a wrong aggregater!!")
            exit(0)
        return self.lambda_a * id_emb + (1 - self.lambda_a) * out

    def user_attention_pooling(self, id_emb, sequence_emb):
        key = self.W_att(sequence_emb) # b x seq_len x attention_dim
        mask = sequence_emb.sum(dim=-1) == 0
        attention = torch.bmm(key, id_emb.unsqueeze(-1)).squeeze(-1) # b x seq_len
        attention = self.masked_softmax(attention, mask)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.bmm(attention.unsqueeze(1), sequence_emb).squeeze(1)
        return self.W_agg(output)

    def mean_pooling(self, sequence_emb):
        mask = sequence_emb.sum(dim=-1) != 0
        mean = sequence_emb.sum(dim=1) / (mask.float().sum(dim=-1, keepdim=True) + 1.e-12)
        return self.W_agg(mean)

    def item_similarity_pooling(self, sequence_emb, score):
        if len(score.size()) != 2:
            score = score.view(score.size(0), -1)
        score = F.softmax(score, dim = -1)
        score = score.unsqueeze(-1)
        ans = (score * sequence_emb).sum(dim=1)
        return self.W_agg(ans)

    def masked_softmax(self, X, mask):
        # use the following softmax to avoid nans when a sequence is entirely masked
        X = X.masked_fill_(mask, 0)
        e_X = torch.exp(X)
        return e_X / (e_X.sum(dim=1, keepdim=True) + 1.e-12)
