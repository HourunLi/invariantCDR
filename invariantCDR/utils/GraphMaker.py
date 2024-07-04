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
        self.UV, self.VU, self.adj, self.ease = self.preprocess(data)
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
    
    def getEASE(self, UV_adj):
        X = copy.deepcopy(UV_adj)
        G = X.T.dot(X).toarray()
        diagIndices = np.diag_indices(G.shape[0])

        G[diagIndices] += self.args.lambda_ease
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[diagIndices] = 0
        EASE_pred = X.dot(B) # the strength of the recommendation of an item to a user.

        def cal_num(X):
            sans = sp.csr_matrix(X)
            print(sans.nnz)
            
        # To accelerate computation, filter some smaller elements
        cal_num(EASE_pred)
        EASE_pred[EASE_pred < 0.1] = 0
        cal_num(EASE_pred)

        
        EASE_pred = sp.csr_matrix(EASE_pred)
        EASE_pred = self._normalize(EASE_pred)
        EASE_pred = self._sparse_mx_to_torch_sparse_tensor(EASE_pred)
        return EASE_pred
    
    def augmentation(self, data):
        
        def takeSecond(elem):
            return -elem[1]
                        
        ease_dense = self.ease.to_dense()
        
        aug_UV_edges = []
        aug_VU_edges = []
        aug_all_edges = []
        
        positive_list, positive_set = {}, {}

        # 添加原本的数据内容
        for edge in data:
            user, item = edge[0],edge[1]
            positive_list[user] = []
            if user not in positive_set.keys():
                positive_set[user] = set()
                positive_list[user] = []
            if item not in positive_set[user]:
                positive_set[user].add(item)
                positive_list[user].append([item, ease_dense[user][item]])

        user_index = self.ease.coalesce().indices()[0].numpy()
        item_index = self.ease.coalesce().indices()[1].numpy()

        # 这里实际上是把所有相似度大于0的item全部加进来了，然后再选择部分数据
        # 既达到了增加数据量的作用也能筛选部分低效数据。
        for id, user in enumerate(user_index):
            item = item_index[id]
            if item not in positive_set[user]:
                positive_list[user].append([item, ease_dense[user][item]])

        def takeSecond(elem):
            return -elem[1]
            
        for user in positive_list:
            positive_list[user].sort(key=takeSecond)
            # To accelerate computation, filter some smaller elements
            positive_list[user] = positive_list[user][:int(1.5 * len(positive_set[user]))]
            for item, score in positive_list[user]:
                if item not in positive_set[user]:
                    positive_set[user].add(item)
        
        total_num = 0
        for user in positive_set:
            items = positive_set[user]
            for item in items:
                total_num = total_num + 1
                aug_UV_edges.append([user, item])
                aug_VU_edges.append([item, user])
                aug_all_edges.append([user,item + self.number_user])
                aug_all_edges.append([item + self.number_user, user])
            
        print("Original edge number: {}; Augmentation edge number: {}; adding {} edges".format(len(data), total_num, total_num - len(data)))
        
        aug_UV_edges = np.array(aug_UV_edges)
        aug_VU_edges = np.array(aug_VU_edges)
        aug_all_edges = np.array(aug_all_edges)
        aug_UV_adj = sp.coo_matrix((np.ones(aug_UV_edges.shape[0]), (aug_UV_edges[:, 0], aug_UV_edges[:, 1])), shape=(self.number_user, self.number_item), dtype=np.float32)
        aug_VU_adj = sp.coo_matrix((np.ones(aug_VU_edges.shape[0]), (aug_VU_edges[:, 0], aug_VU_edges[:, 1])), shape=(self.number_item, self.number_user), dtype=np.float32)
        aug_all_adj = sp.coo_matrix((np.ones(aug_all_edges.shape[0]), (aug_all_edges[:, 0], aug_all_edges[:, 1])),shape=(self.number_item+self.number_user, self.number_item+self.number_user),dtype=np.float32)

        aug_UV_adj = self._normalize(aug_UV_adj)
        aug_VU_adj = self._normalize(aug_VU_adj)
        aug_UV_adj = self._sparse_mx_to_torch_sparse_tensor(aug_UV_adj).to(self.args.device)
        aug_VU_adj = self._sparse_mx_to_torch_sparse_tensor(aug_VU_adj).to(self.args.device)
        
        aug_all_adj = self._normalize(aug_all_adj)
        if self.args.A_split:
            aug_all_adj = self._split_A_hat(aug_all_adj)
            print("done split matrix")
        else:
            aug_all_adj = self._sparse_mx_to_torch_sparse_tensor(aug_all_adj)
            aug_all_adj = aug_all_adj.coalesce().to(self.args.device)
            
        print("augmentation graph loaded!")
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

        # tempg = torch.sparse_coo_tensor(torch.LongTensor([all_adj.row, all_adj.col]),
        #                                 torch.FloatTensor(all_adj.data),
        #                                 torch.Size(all_adj.shape)).coalesce()
        # deg = torch.sparse.sum(tempg, dim=1).to_dense()
        # deg = torch.where(deg == 0, torch.tensor(1.0), deg).to(self.args.device)
        
        # get ease similarity
        EASE_pred = self.getEASE(UV_adj)
            
        UV_adj = self._normalize(UV_adj)
        VU_adj = self._normalize(VU_adj)
        # all_adj = normalize(all_adj)
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
        return UV_adj, VU_adj, all_adj, EASE_pred


    # def augProcess(self, res_edges):
    #     aug_UV_edges = []
    #     aug_VU_edges = []
    #     all_aug_edges = []
    #     aug_real_adj = {}

        
    #     for edge in res_edges:
    #         aug_UV_edges.append([edge[0],edge[1]])
    #         aug_VU_edges.append([edge[1], edge[0]])

    #         all_aug_edges.append([edge[0],edge[1] + self.number_user])
    #         all_aug_edges.append([edge[1] + self.number_user, edge[0]])
    #         if edge[0] not in aug_real_adj:
    #             aug_real_adj[edge[0]] = {}
    #         aug_real_adj[edge[0]][edge[1]] = 1

    #     aug_UV_edges = np.array(aug_UV_edges)
    #     aug_VU_edges = np.array(aug_VU_edges)
    #     all_aug_edges = np.array(all_aug_edges)
        
    #     aug_UV_adj = sp.coo_matrix((np.ones(aug_UV_edges.shape[0]), (aug_UV_edges[:, 0], aug_UV_edges[:, 1])), shape=(self.number_user, self.number_item), dtype=np.float32)
    #     aug_VU_adj = sp.coo_matrix((np.ones(aug_VU_edges.shape[0]), (aug_VU_edges[:, 0], aug_VU_edges[:, 1])), shape=(self.number_item, self.number_user), dtype=np.float32)
    #     aug_all_adj = sp.coo_matrix((np.ones(all_aug_edges.shape[0]), (all_aug_edges[:, 0], all_aug_edges[:, 1])),shape=(self.number_item+self.number_user, self.number_item + self.number_user),dtype=np.float32)
        
    #     aug_UV_adj = self._normalize(aug_UV_adj)
    #     aug_VU_adj = self._normalize(aug_VU_adj)
    #     # all_adj = normalize(all_adj)
    #     aug_UV_adj = self._sparse_mx_to_torch_sparse_tensor(aug_UV_adj).to(self.args.device)
    #     aug_VU_adj = self._sparse_mx_to_torch_sparse_tensor(aug_VU_adj).to(self.args.device)
        
    #     aug_all_adj = self._normalize_lightgcn(aug_all_adj)
    #     if self.args.A_split:
    #         aug_all_adj = self._split_A_hat(aug_all_adj)
    #         print("done split augmented matrix")
    #     else:
    #         aug_all_adj = self._sparse_mx_to_torch_sparse_tensor(aug_all_adj)
    #         aug_all_adj = aug_all_adj.coalesce().to(self.args.device)
            
    #     # print("augmented graph loaded!")
    #     return aug_UV_adj, aug_VU_adj, aug_all_adj
    
    # def getAugmentedGraph(self, user_emb_, item_emb_, ratio=0.1):
    #     # print("=====Aug_edges begin=====")
    #     aug_train_User = list(copy.deepcopy(self.raw_data[:, 0]))
    #     aug_train_Item = list(copy.deepcopy(self.raw_data[:, 1]))
    #     # print(len(aug_train_User), len(aug_train_Item))
    #     # normalization
        
    #     deg1 = self.deg
    #     deg_cpu = self.deg.clone().cpu()
    #     user_emb = user_emb_.detach()
    #     item_emb = item_emb_.detach()
    #     B, K, d = user_emb.size()
    #     user_emb = user_emb.view(B, K*d)
    #     B, K, d = item_emb.size()
    #     item_emb = item_emb.view(B, K*d)
    #     user_emb = F.normalize(user_emb, p=2, dim=1)
    #     item_emb = F.normalize(item_emb, p=2, dim=1)
    #     # print(user_emb.size()[0], self.number_user)
    #     assert user_emb.size()[0] == self.number_user
    #     user_emb = user_emb / deg1[: self.number_user].view(-1, 1)
    #     item_emb = item_emb / deg1[self.number_user :].view(-1, 1)

    #     n = user_emb.size(0)
    #     sample_size = int(self.number_item * ratio)
    #     indices = torch.randperm(self.number_item)[:sample_size].cuda()
    #     sampled_item_embeddings = item_emb[indices]

    #     nearest_distances = torch.full((n,), float("inf"), device = self.args.device)
    #     nearest_indices = torch.zeros(n, dtype=torch.long, device = self.args.device)
    #     batch_size = 2048
        
    #     with torch.no_grad():
    #         # for i in tqdm(range(0, n, batch_size)):
    #         for i in range(0, n, batch_size):
    #             user_batch = user_emb[i : i + batch_size].unsqueeze(1)  # Shape: (batch_size, 1, d)
    #             items = sampled_item_embeddings.unsqueeze(0)  # Shape: (1, sample_size, d)
    #             distances = (user_batch - items).norm(dim=2)  # Shape: (batch_size, sample_size)
    #             # print("distances size: {}".format(distances.size()))
    #             min_distances, min_indices = distances.min(dim=1)
    #             # print(nearest_distances[i : i + batch_size])
    #             nearest_indices[i : i + batch_size] = torch.where(
    #                 min_distances < nearest_distances[i : i + batch_size],
    #                 indices[min_indices],
    #                 nearest_indices[i : i + batch_size],
    #             )
    #             nearest_distances[i : i + batch_size] = torch.min(nearest_distances[i : i + batch_size], min_distances)

    #     # print(nearest_indices.size())
    #     # print(len(aug_train_User), len(aug_train_Item))
    #     aug_train_Item.extend(nearest_indices.tolist())
    #     aug_train_User.extend(range(self.number_user))
    #     # print(len(aug_train_User), len(aug_train_Item))
        
    #     unique_pairs = set()
    #     unique_indices = []
    #     for idx, (user, item) in enumerate(zip(aug_train_User, aug_train_Item)):
    #         if (user, item) not in unique_pairs:
    #             unique_pairs.add((user, item))
    #             unique_indices.append(idx)

    #     aug_train_User = [aug_train_User[i] for i in unique_indices]
    #     aug_train_Item = [aug_train_Item[i] for i in unique_indices]

    #     res_edges = np.stack((aug_train_User, aug_train_Item), axis=1)
    #     # print(res_edges.shape)
    #     # res_edges = np.array(unique_indices)
    #     # print(res_edges.shape)
    #     self.aug_UV, self.aug_VU, self.aug_adj =  self.augProcess(res_edges)
    #     self.aug_res_edges = res_edges
    #     # print("New add edges num: ", res_edges.shape[0] - len(self.raw_data))
    #     # print("======Aug_edges done!======")
    #     return self.aug_UV, self.aug_VU
