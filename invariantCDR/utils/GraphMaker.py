import numpy as np
import random
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import codecs
import json
import copy
from tqdm import tqdm

def convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

class GraphMaker(object):
    def __init__(self, args, filename):
        self.args = args
        self.user = set()
        self.item = set()
        self.folds = args.a_fold
        data=[]
        with codecs.open(filename) as infile:
            for line in infile:
                line = line.strip().split("\t")
                data.append([int(line[0]), int(line[1])])
                self.user.add(int(line[0]))
                self.item.add(int(line[1]))

        self.number_user = max(self.user) + 1
        self.number_item = max(self.item) + 1
        print("number_user", self.number_user)
        print("number_item", self.number_item)
        
        self.raw_data = np.array(data)
        self.UV, self.VU, self.adj = self.preprocess(data)

    def _normalize(self, mx):
        # print(f"mx shape is {mx.shape}")
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        # print(f"row sum shape is {rowsum.shape}")
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        # print(f"r_inv shape is {r_inv.shape}")
        r_mat_inv = sp.diags(r_inv)
        # print(f"r_mat_inv shape is {r_mat_inv.shape}")
        mx = r_mat_inv.dot(mx)
        # print(f"return shape is {mx.shape}")
        return mx

    def _normalize_lightgcn(self, mx):
        rowsum = np.array(mx.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        
        norm_adj = d_mat.dot(mx)
        norm_adj = norm_adj.dot(d_mat)
        return norm_adj

    def _sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.number_user + self.number_item) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.number_user + self.number_item
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._sparse_mx_to_torch_sparse_tensor(A[start:end]).coalesce().to(self.args.device))
        return A_fold
        
    def preprocess(self,data):
        UV_edges = []
        VU_edges = []
        all_edges = []
        
        for edge in data:
            UV_edges.append([edge[0],edge[1]])
            VU_edges.append([edge[1], edge[0]])
            all_edges.append([edge[0],edge[1] + self.number_user])
            all_edges.append([edge[1] + self.number_user, edge[0]])

        UV_edges = np.array(UV_edges)
        VU_edges = np.array(VU_edges)
        all_edges = np.array(all_edges)
        UV_adj = sp.coo_matrix((np.ones(UV_edges.shape[0]), (UV_edges[:, 0], UV_edges[:, 1])), shape=(self.number_user, self.number_item), dtype=np.float32)
        VU_adj = sp.coo_matrix((np.ones(VU_edges.shape[0]), (VU_edges[:, 0], VU_edges[:, 1])), shape=(self.number_item, self.number_user), dtype=np.float32)
        all_adj = sp.coo_matrix((np.ones(all_edges.shape[0]), (all_edges[:, 0], all_edges[:, 1])),shape=(self.number_item+self.number_user, self.number_item+self.number_user),dtype=np.float32)
            
        UV_adj = self._normalize(UV_adj)
        VU_adj = self._normalize(VU_adj)
        UV_adj = self._sparse_mx_to_torch_sparse_tensor(UV_adj).to(self.args.device)
        VU_adj = self._sparse_mx_to_torch_sparse_tensor(VU_adj).to(self.args.device)

        all_adj = self._normalize(all_adj)
        if self.args.A_split:
            all_adj = self._split_A_hat(all_adj)
            print("done split matrix")
        else:
            all_adj = self._sparse_mx_to_torch_sparse_tensor(all_adj)
            all_adj = all_adj.coalesce().to(self.args.device)
            
        print("real graph loaded!")
        return UV_adj, VU_adj, all_adj
        