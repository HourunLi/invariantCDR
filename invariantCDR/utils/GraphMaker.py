import numpy as np
import random
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import codecs
import json
import copy
from tqdm import tqdm
import cupy as cp

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
        # self.UV, self.VU, self.adj, self.ease = self.preprocess(data)
        self.UV, self.VU, self.adj = self.preprocess(data)
        self.aug_UV, self.aug_VU, self.aug_adj = self.augmentation(data)

    def _normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
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
    
    # def getEASE(self, UV_adj):
    #     X = UV_adj.coalesce()  # 确保 UV_adj 是 coalesced
    #     G = torch.sparse.mm(X.transpose(0, 1), X)

    #     # 添加 lambda_ease 到对角线
    #     lambda_ease = self.args.lambda_ease
    #     identity = torch.eye(G.shape[0]).to(G.device) * lambda_ease
    #     G = G + identity.to_sparse()

    #     # 使用共轭梯度法求解线性系统 Gx = I（等效于求G的逆）
    #     I = torch.eye(G.shape[0], device=G.device)
    #     G_dense = G.to_dense()  # 仅用于初始化和比较
    #     x_sol = torch.linalg.solve(G_dense, I)  # 使用稠密矩阵求解线性系统

    #     # 用解 x_sol 替代直接计算的逆矩阵
    #     B = -x_sol / torch.diag(x_sol)[:, None]
    #     B[torch.arange(G.shape[0]), torch.arange(G.shape[0])] = 0

    #     EASE_pred = torch.mm(X.to_dense(), B)
    #     def cal_num(X):
    #         # 使用 PyTorch 计算非零元素
    #         print((X != 0).sum().item())
        
    #     # 过滤较小元素
    #     cal_num(EASE_pred)
    #     EASE_pred[EASE_pred < 0.1] = 0
    #     cal_num(EASE_pred)
    #     EASE_pred = EASE_pred.to_sparse()
    #     return EASE_pred
    #     # X = sp.csr_matrix(UV_adj)
    #     # G = X.T.dot(X)
    #     # diagIndices = cp.diag_indices(G.shape[0])

    #     # G[diagIndices] += self.args.lambda_ease
    #     # P = cp.linalg.inv(G)
    #     # B = P / (-cp.diag(P))
    #     # B[diagIndices] = 0
    #     # EASE_pred = X.dot(B) # the strength of the recommendation of an item to a user.

    #     # def cal_num(X):
    #     #     sans = sp.csr_matrix(X)
    #     #     print(sans.nnz)
            
    #     # # To accelerate computation, filter some smaller elements
    #     # cal_num(EASE_pred)
    #     # EASE_pred[EASE_pred < 0.1] = 0
    #     # cal_num(EASE_pred)

        
    #     # EASE_pred = sp.csr_matrix(cp.asnumpy(EASE_pred))
    #     # EASE_pred = self._normalize(EASE_pred)
    #     # EASE_pred = self._sparse_mx_to_torch_sparse_tensor(EASE_pred)
    #     return EASE_pred
    
    # def augmentation(self, data):
        
    #     def takeSecond(elem):
    #         return -elem[1]
                        
    #     ease_dense = self.ease.to_dense()
        
    #     aug_UV_edges = []
    #     aug_VU_edges = []
    #     aug_all_edges = []
        
    #     positive_list, positive_set = {}, {}

    #     # 添加原本的数据内容
    #     for edge in data:
    #         user, item = edge[0],edge[1]
    #         positive_list[user] = []
    #         if user not in positive_set.keys():
    #             positive_set[user] = set()
    #             positive_list[user] = []
    #         if item not in positive_set[user]:
    #             positive_set[user].add(item)
    #             positive_list[user].append([item, ease_dense[user][item]])

    #     user_index = self.ease.coalesce().indices()[0].numpy()
    #     item_index = self.ease.coalesce().indices()[1].numpy()

    #     # 这里实际上是把所有相似度大于0的item全部加进来了，然后再选择部分数据
    #     # 既达到了增加数据量的作用也能筛选部分低效数据。
    #     for id, user in enumerate(user_index):
    #         item = item_index[id]
    #         if item not in positive_set[user]:
    #             positive_list[user].append([item, ease_dense[user][item]])

    #     def takeSecond(elem):
    #         return -elem[1]
            
    #     for user in positive_list:
    #         positive_list[user].sort(key=takeSecond)
    #         # To accelerate computation, filter some smaller elements
    #         positive_list[user] = positive_list[user][:int(1.5 * len(positive_set[user]))]
    #         for item, score in positive_list[user]:
    #             if item not in positive_set[user]:
    #                 positive_set[user].add(item)
        
    #     total_num = 0
    #     for user in positive_set:
    #         items = positive_set[user]
    #         for item in items:
    #             total_num = total_num + 1
    #             aug_UV_edges.append([user, item])
    #             aug_VU_edges.append([item, user])
    #             aug_all_edges.append([user,item + self.number_user])
    #             aug_all_edges.append([item + self.number_user, user])
            
    #     print("Original edge number: {}; Augmentation edge number: {}; adding {} edges".format(len(data), total_num, total_num - len(data)))
        
    #     aug_UV_edges = np.array(aug_UV_edges)
    #     aug_VU_edges = np.array(aug_VU_edges)
    #     aug_all_edges = np.array(aug_all_edges)
    #     aug_UV_adj = sp.coo_matrix((np.ones(aug_UV_edges.shape[0]), (aug_UV_edges[:, 0], aug_UV_edges[:, 1])), shape=(self.number_user, self.number_item), dtype=np.float32)
    #     aug_VU_adj = sp.coo_matrix((np.ones(aug_VU_edges.shape[0]), (aug_VU_edges[:, 0], aug_VU_edges[:, 1])), shape=(self.number_item, self.number_user), dtype=np.float32)
    #     aug_all_adj = sp.coo_matrix((np.ones(aug_all_edges.shape[0]), (aug_all_edges[:, 0], aug_all_edges[:, 1])),shape=(self.number_item+self.number_user, self.number_item+self.number_user),dtype=np.float32)

    #     aug_UV_adj = self._normalize(aug_UV_adj)
    #     aug_VU_adj = self._normalize(aug_VU_adj)
    #     aug_UV_adj = self._sparse_mx_to_torch_sparse_tensor(aug_UV_adj).to(self.args.device)
    #     aug_VU_adj = self._sparse_mx_to_torch_sparse_tensor(aug_VU_adj).to(self.args.device)
        
    #     aug_all_adj = self._normalize(aug_all_adj)
    #     if self.args.A_split:
    #         aug_all_adj = self._split_A_hat(aug_all_adj)
    #         print("done split matrix")
    #     else:
    #         aug_all_adj = self._sparse_mx_to_torch_sparse_tensor(aug_all_adj)
    #         aug_all_adj = aug_all_adj.coalesce().to(self.args.device)
            
    #     print("augmentation graph loaded!")
    #     return aug_UV_adj, aug_VU_adj, aug_all_adj

    def augmentation(self, data):
        aug_UV_edges = []
        aug_VU_edges = []
        aug_all_edges = []

        data = np.array(data)
        num_edges = data.shape[0]
        num_selected_edges = int(num_edges * 0.9)
        all_indices = list(range(num_edges))
        selected_indices = random.sample(all_indices, num_selected_edges)
        selected_edges = data[selected_indices]

        for edge in selected_edges:
            aug_UV_edges.append([edge[0],edge[1]])
            aug_VU_edges.append([edge[1], edge[0]])
            aug_all_edges.append([edge[0],edge[1] + self.number_user])
            aug_all_edges.append([edge[1] + self.number_user, edge[0]])

        aug_UV_edges = np.array(aug_UV_edges)
        aug_VU_edges = np.array(aug_VU_edges)
        aug_all_edges = np.array(aug_all_edges)
        aug_UV_adj = sp.coo_matrix((np.ones(aug_UV_edges.shape[0]), (aug_UV_edges[:, 0], aug_UV_edges[:, 1])), shape=(self.number_user, self.number_item), dtype=np.float32)
        aug_VU_adj = sp.coo_matrix((np.ones(aug_VU_edges.shape[0]), (aug_VU_edges[:, 0], aug_VU_edges[:, 1])), shape=(self.number_item, self.number_user), dtype=np.float32)
        aug_all_adj = sp.coo_matrix((np.ones(aug_all_edges.shape[0]), (aug_all_edges[:, 0], aug_all_edges[:, 1])),shape=(self.number_item+self.number_user, self.number_item+self.number_user),dtype=np.float32)
            
        aug_UV_adj = self._normalize(aug_UV_adj)
        aug_VU_adj = self._normalize(aug_VU_adj)
        # aug_all_adj = normalize(aug_all_adj)
        aug_UV_adj = self._sparse_mx_to_torch_sparse_tensor(aug_UV_adj).to(self.args.device)
        aug_VU_adj = self._sparse_mx_to_torch_sparse_tensor(aug_VU_adj).to(self.args.device)

        aug_all_adj = self._normalize(aug_all_adj)
        if self.args.A_split:
            aug_all_adj = self._split_A_hat(aug_all_adj)
            print("done split matrix")
        else:
            aug_all_adj = self._sparse_mx_to_torch_sparse_tensor(aug_all_adj)
            aug_all_adj = aug_all_adj.coalesce().to(self.args.device)
            
        print("augmented graph loaded!")
        return aug_UV_adj, aug_VU_adj, aug_all_adj
        
        
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
        # all_adj = normalize(all_adj)
        UV_adj = self._sparse_mx_to_torch_sparse_tensor(UV_adj).to(self.args.device)
        VU_adj = self._sparse_mx_to_torch_sparse_tensor(VU_adj).to(self.args.device)

        # get ease similarity
        # EASE_pred = self.getEASE(UV_adj)
        
        all_adj = self._normalize(all_adj)
        if self.args.A_split:
            all_adj = self._split_A_hat(all_adj)
            print("done split matrix")
        else:
            all_adj = self._sparse_mx_to_torch_sparse_tensor(all_adj)
            all_adj = all_adj.coalesce().to(self.args.device)
            
        print("real graph loaded!")
        # return UV_adj, VU_adj, all_adj, EASE_pred
        return UV_adj, VU_adj, all_adj
        