import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import math
from copy import deepcopy
from itertools import repeat
from torch_geometric.data import DataLoader, InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data
from torch_geometric.utils import degree
from torch.nn import Sequential, Linear, ReLU
from torch.autograd import Variable
from model.GCN import LightGCN
from model.Encoder import VBGE, disenEncoder
from model.transfer import FactorDomainTransformer
from utils import torch_utils, helper

class invariantCDR(nn.Module):
    def __init__(self, args, device):
        super(invariantCDR, self).__init__()
        self.args = args
        self.device = device
        self.sim_s = None
        self.sim_t = None
        # self.source_LightGCN = LightGCN(args, args.source_user_num, args.source_item_num, args.source_adj)
        # self.target_LightGCN = LightGCN(args, args.target_user_num, args.target_item_num, args.target_adj)
        self.source_disenEncoder = disenEncoder(args)
        self.target_disenEncoder = disenEncoder(args)
        
        self.discri_source = nn.Sequential(
            nn.Linear(args.feature_dim * 2, args.feature_dim),
            nn.LeakyReLU(args.leakey),
            nn.Linear(args.feature_dim, 100),
            nn.LeakyReLU(args.leakey),
            nn.Linear(100, 1),
        )
        self.discri_target = nn.Sequential(
            nn.Linear(args.feature_dim * 2, args.feature_dim),
            nn.LeakyReLU(args.leakey),
            nn.Linear(args.feature_dim, 100),
            nn.LeakyReLU(args.leakey),
            nn.Linear(100, 1),
        )
        
        self.s2t_transfer = FactorDomainTransformer(args)
        self.t2s_transfer = FactorDomainTransformer(args)
        
        self.tau = args.tau 
        self.source_user_embedding = nn.Embedding(args.source_user_num, args.feature_dim).to(self.device)
        self.target_user_embedding = nn.Embedding(args.target_user_num, args.feature_dim).to(self.device)
        self.source_item_embedding = nn.Embedding(args.source_item_num, args.feature_dim).to(self.device)
        self.target_item_embedding = nn.Embedding(args.target_item_num, args.feature_dim).to(self.device)

        self.source_user_index = torch.arange(0, self.args.source_user_num, 1).to(self.device)
        self.target_user_index = torch.arange(0, self.args.target_user_num, 1).to(self.device)
        self.source_item_index = torch.arange(0, self.args.source_item_num, 1).to(self.device)
        self.target_item_index = torch.arange(0, self.args.target_item_num, 1).to(self.device)
        
        self.criterion = nn.BCEWithLogitsLoss().to(self.device)
        self.transfer_criterion = nn.MSELoss()
        self.critic_loss = 0
        self.transfer_flag = 0
        self.K = args.num_latent_factors 
        self.d = args.feature_dim // self.K
        
        self.center = torch.rand((self.K, self.d), requires_grad=True).to(self.device)
        self.source_user_center = torch.rand((self.K, self.d), requires_grad=True).to(self.device)
        self.source_item_center = torch.rand((self.K, self.d), requires_grad=True).to(self.device)
        self.target_user_center = torch.rand((self.K, self.d), requires_grad=True).to(self.device)
        self.target_item_center = torch.rand((self.K, self.d), requires_grad=True).to(self.device)
        
        self.dropout = self.args.dropout
        self.__init_weight()
        
    def __init_weight(self):
        nn.init.normal_(self.source_user_embedding.weight, std=0.1)
        nn.init.normal_(self.target_user_embedding.weight, std=0.1)     
        nn.init.normal_(self.source_item_embedding.weight, std=0.1)
        nn.init.normal_(self.target_item_embedding.weight, std=0.1)

    def source_predict_nn(self, user_embedding, item_embedding):
        fea = torch.cat((user_embedding, item_embedding), dim=-1)
        out = self.source_predict_1(fea)
        out = F.relu(out)
        out = self.source_predict_2(out)
        out = torch.sigmoid(out)
        return out

    def target_predict_nn(self, user_embedding, item_embedding):
        fea = torch.cat((user_embedding, item_embedding), dim=-1)
        out = self.target_predict_1(fea)
        out = F.relu(out)
        out = self.target_predict_2(out)
        out = torch.sigmoid(out)
        return out

    def source_predict_dot(self, user_embedding, item_embedding):
        output = (user_embedding * item_embedding).sum(dim=-1)
        # return torch.sigmoid(output)
        return output

    def target_predict_dot(self, user_embedding, item_embedding):
        output = (user_embedding * item_embedding).sum(dim=-1)
        # return torch.sigmoid(output)
        return output

    def user_similarity_dot(self, user_embedding_A, user_embedding_B):
        output = (user_embedding_A * user_embedding_B).sum(dim=-1)
        # return torch.sigmoid(output)
        return output
    
    def my_index_select(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = torch.index_select(memory, 0, index)
        ans = ans.view(tmp)
        return ans
    
    def dis(self, A, B):
        return self.user_similarity_dot(A, B)
        # C = torch.cat((A,B), dim = 1)
        # return self.discri(C)
    
    def HingeLoss(self, pos, neg):
        pos = F.sigmoid(pos)
        neg = F.sigmoid(neg)
        gamma = torch.tensor(self.args.margin)
        if self.args.cuda:
            gamma = gamma.cuda()
        return F.relu(gamma - pos + neg).mean()

    # def getRegLoss(self, source_user_specific, source_pos_item, source_neg_item, target_user_specific, target_pos_item, target_neg_item):
    #     source_users_emb_ego = self.source_user_embedding_specific(source_user_specific)
    #     source_pos_emb_ego = self.source_item_embedding(source_pos_item)
    #     source_neg_emb_ego = self.source_item_embedding(source_neg_item)
        
    #     target_users_emb_ego = self.target_user_embedding_specific(target_user_specific)
    #     target_pos_emb_ego = self.target_item_embedding(target_pos_item)
    #     target_neg_emb_ego = self.target_item_embedding(target_neg_item)
        
    #     source_reg_loss = (1/2)*(source_users_emb_ego.norm(2).pow(2) + 
    #                      source_pos_emb_ego.norm(2).pow(2)  +
    #                      source_neg_emb_ego.norm(2).pow(2))/float(len(source_user_specific))
        
    #     target_reg_loss = (1/2)*(target_users_emb_ego.norm(2).pow(2) + 
    #                      target_pos_emb_ego.norm(2).pow(2)  +
    #                      target_neg_emb_ego.norm(2).pow(2))/float(len(target_user_specific))
        
    #     return source_reg_loss + target_reg_loss

    def min_max_norm(self, embeddings):
        min_vals = embeddings.min(dim=1, keepdim=True)[0]
        max_vals = embeddings.max(dim=1, keepdim=True)[0]
        embeddings = (embeddings - min_vals) / (max_vals - min_vals)
        return embeddings
    
    def row_wise_norm(self, embeddings):
        row_sums = embeddings.sum(dim=1, keepdim=True)
        embeddings = embeddings / row_sums
        return embeddings

    def _cal_similarity_matrix(self, embeddings):
        norm_embeddings = F.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.mm(norm_embeddings, norm_embeddings.t())
        # binary_similarity_matrix = (similarity_matrix > self.args.sim_threshold).float()
        # return similarity_matrix
        # print("-----------source_sim_matrices------------")
        # print(binary_similarity_matrix)
        exp_sim = torch.exp(similarity_matrix / self.tau)
        exp_sim = self.min_max_norm(exp_sim)
        exp_sim = self.row_wise_norm(exp_sim)
        exp_sim[exp_sim < 0.2] = 0
        return exp_sim

    def cal_similarity_matrices(self, source_learn_user, target_learn_user):
        source_sim_matrices = []
        target_sim_matrices = []
        source_shared_user = source_learn_user[:self.args.test_user].to(self.device)
        target_shared_user = target_learn_user[:self.args.test_user].to(self.device)

        for i in range(self.K):
            source_factor_embedding = source_shared_user[:, i, :]
            target_factor_embedding = target_shared_user[:, i, :]
            source_sim = self._cal_similarity_matrix(source_factor_embedding).detach()
            target_sim = self._cal_similarity_matrix(target_factor_embedding).detach()
            source_sim_matrices.append(source_sim)
            target_sim_matrices.append(target_sim)
            # print(sim_source.size())
            # print(sim_target.size())
        # print("-----------source_sim_matrices------------")
        # print(source_sim_matrices[0])
        # print(source_sim_matrices[1])
        
        # print("-----------target_sim_matrices------------")
        # print(target_sim_matrices)
        return source_sim_matrices, target_sim_matrices
    
    def _cal_transfer_loss(self, embeddings, similarity):
        transfer_sim = self._cal_similarity_matrix(embeddings)
        loss = self.transfer_criterion(transfer_sim, similarity)
        return loss
    
    def cal_transfer_loss(self, source_learn_user_transfer, target_learn_user_transfer, source_similarity_batch, target_similarity_batch):
        source_sim_loss = 0
        target_sim_loss = 0
        for i in range(self.K):
            source_sim_loss += self._cal_transfer_loss(source_learn_user_transfer[:, i, :], source_similarity_batch[i])
            target_sim_loss += self._cal_transfer_loss(target_learn_user_transfer[:, i, :], target_similarity_batch[i])
        return source_sim_loss + target_sim_loss
    
    def forward(self, source_UV, source_VU, target_UV, target_VU):
        source_user = self.source_user_embedding(self.source_user_index)
        target_user = self.target_user_embedding(self.target_user_index)
        source_item = self.source_item_embedding(self.source_item_index)
        target_item = self.target_item_embedding(self.target_item_index)

        # source_learn_user, source_learn_item = self.source_GNN(source_user, source_item, source_UV, source_VU)
        # target_learn_user, target_learn_item = self.target_GNN(target_user, target_item, target_UV, target_VU)        
        
        source_learn_user, source_learn_item = self.source_disenEncoder(source_user, source_item, source_UV, source_VU)
        target_learn_user, target_learn_item = self.target_disenEncoder(target_user, target_item, target_UV, target_VU)
        source_learn_user_concat, target_learn_user_concat = None, None
        
        if self.transfer_flag == 0:
            # shape: (B, K, D)
            source_learn_user_concat =  source_learn_user
            target_learn_user_concat =  target_learn_user
            self.critic_loss = 0
            return source_learn_user_concat, source_learn_item, target_learn_user_concat, target_learn_item
        
        if self.training:
            # 根据第一阶段的相似度，用于transfer guideline
            if self.sim_s is None or self.sim_t is None:
                self.sim_s, self.sim_t = self.cal_similarity_matrices(source_learn_user, target_learn_user)
            
            # calculate critic loss
            per_stable = torch.randperm(self.args.shared_user)[:self.args.user_batch_size].to(self.device)
            source_learn_user_stable = source_learn_user[per_stable]
            target_learn_user_stable = target_learn_user[per_stable]
            source_learn_user_transfer = self.t2s_transfer.forward_user(target_learn_user_stable)
            target_learn_user_transfer = self.s2t_transfer.forward_user(source_learn_user_stable)
            source_similarity_matrix = [self.sim_s[i][per_stable, :][:, per_stable] for i in range(self.K)]
            target_similarity_matrix = [self.sim_t[i][per_stable, :][:, per_stable] for i in range(self.K)]
            critic_loss = self.cal_transfer_loss(source_learn_user_transfer, target_learn_user_transfer, source_similarity_matrix, target_similarity_matrix)
            self.critic_loss = critic_loss
            
            # calculate transfer embeddings
            transfer_source_user_embedding = self.t2s_transfer(target_learn_user[:self.args.shared_user], self.sim_t, source_learn_item, source_UV)
            transfer_target_user_embedding = self.s2t_transfer(source_learn_user[:self.args.shared_user], self.sim_s, target_learn_item, target_UV)
            source_learn_user_concat = torch.cat((transfer_source_user_embedding, source_learn_user[self.args.shared_user:]),dim=0)
            target_learn_user_concat = torch.cat((transfer_target_user_embedding, target_learn_user[self.args.shared_user:]),dim=0)
        else :
            transfer_source_user_embedding = self.t2s_transfer(target_learn_user[:self.args.source_shared_user], self.sim_t, source_learn_item, source_UV)
            transfer_target_user_embedding = self.s2t_transfer(source_learn_user[:self.args.target_shared_user], self.sim_s, target_learn_item, target_UV)
            source_learn_user_concat = torch.cat((transfer_source_user_embedding, source_learn_user[self.args.source_shared_user:]), dim=0)
            target_learn_user_concat = torch.cat((transfer_target_user_embedding, target_learn_user[self.args.target_shared_user:]), dim=0)

        return source_learn_user_concat, source_learn_item, target_learn_user_concat, target_learn_item
    