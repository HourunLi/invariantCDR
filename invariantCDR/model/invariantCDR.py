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
from model.Encoder import VBGE
from model.Encoder import disenEncoder
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
        
        self.source2target =  nn.Sequential(
            nn.Linear(args.feature_dim, args.feature_dim),
            nn.LeakyReLU(args.leakey),
            nn.Linear(args.feature_dim, args.feature_dim),
        )
        self.target2source =  nn.Sequential(
            nn.Linear(args.feature_dim, args.feature_dim),
            nn.LeakyReLU(args.leakey),
            nn.Linear(args.feature_dim, args.feature_dim),
        )
        # self.source_GNN = VBGE(args)
        # self.target_GNN = VBGE(args)
        # self.discri = nn.Sequential(
        #     nn.Linear(args.feature_dim * 2 * self.args.GNN, args.feature_dim),
        #     nn.ReLU(),
        #     nn.Linear(args.feature_dim, 100),
        #     nn.ReLU(),
        #     nn.Linear(100, 1),
        # )
        # The temperature parameter for contrastive learning, set to 0.2
        
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
        self.critic_loss = 0
        self.transfer_flag = 0
        self.K = args.num_latent_factors 
        self.d = args.feature_dim // self.K
        # self.center_v = torch.rand((self.K, self.d), requires_grad=True).to(self.device)
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
        
    def cosine_similarity_matrix(self, embeddings):
        norm_embeddings = F.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.mm(norm_embeddings, norm_embeddings.t())
        binary_similarity_matrix = (similarity_matrix > self.args.sim_threshold).float()
        exp_sim = torch.exp(similarity_matrix / self.tau)
        softmax_similarity_matrix = F.softmax(exp_sim, dim=1)
        softmax_similarity_matrix = softmax_similarity_matrix.detach()
        return softmax_similarity_matrix

    def calculate_sim(self, source_learn_user, target_learn_user):
        source_shared_user = self.my_index_select(source_learn_user, torch.arange(0, self.args.shared_user).to(self.device))
        target_shared_user = self.my_index_select(target_learn_user, torch.arange(0, self.args.shared_user).to(self.device))
        sim_s = self.cosine_similarity_matrix(source_shared_user)
        sim_t = self.cosine_similarity_matrix(target_shared_user)
        return sim_s, sim_t
    
    def _transfer_sim(self, embedding, similarity):
        norm_embeddings = F.normalize(embedding, p=2, dim=1)
        # print(embedding)
        sim = torch.mm(norm_embeddings, norm_embeddings.t()) / self.tau
        exp_sim = torch.exp(sim)
        sum_exp_sim = torch.sum(exp_sim, dim=1, keepdim=True)
        exp_sim_positive = exp_sim * similarity
        sum_exp_sim_positive = torch.sum(exp_sim_positive, dim=1).clamp(min=1e-9)
        # print("sum_exp_sim_positive: {}".format(sum_exp_sim_positive.size()))
        # print(sum_exp_sim_positive)
        # print("sum_exp_sim: {}".format(sum_exp_sim.size()))
        # print(sum_exp_sim)
        sim_loss = -torch.log(sum_exp_sim_positive / sum_exp_sim)
        return sim_loss.mean()
    
    def transfer_sim(self, source_learn_user_transfer, target_learn_user_transfer, source_similarity_batch, target_similarity_batch):
        source_sim_loss = self._transfer_sim(source_learn_user_transfer, source_similarity_batch)
        target_sim_loss = self._transfer_sim(target_learn_user_transfer, target_similarity_batch)
        # print(source_sim_loss, target_sim_loss)
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
    
        if self.training:
            if self.transfer_flag == 0:
                # warm up 不做transfer，首先学好每个领域自己内部的知识
                source_learn_user_ret = source_learn_user
                target_learn_user_ret = target_learn_user
                self.critic_loss = 0
            else:
                if self.sim_s is None or self.sim_t is None:
                    self.sim_s, self.sim_t = self.calculate_sim(source_learn_user, target_learn_user)
                    # print(self.sim_s)
                    # print(self.sim_t)
                
                per_stable = torch.randperm(self.args.shared_user)[:self.args.user_batch_size].to(self.device)
                source_learn_user_stable = self.my_index_select(source_learn_user, per_stable)
                target_learn_user_stable = self.my_index_select(target_learn_user, per_stable)
                source_learn_user_transfer = self.target2source(target_learn_user_stable)
                target_learn_user_transfer = self.source2target(source_learn_user_stable)
                # pos_1 = self.dis(target_learn_user_transfer, target_learn_user_stable).view(-1)
                # pos_2 = self.dis(source_learn_user_stable, source_learn_user_transfer).view(-1)
                
                # per = torch.randperm(self.args.target_user_num)[:self.args.user_batch_size].to(self.device)
                # neg_1 = self.dis(target_learn_user_transfer, self.my_index_select(target_learn_user, per)).view(-1)
                
                # per = torch.randperm(self.args.source_user_num)[:self.args.user_batch_size].to(self.device)
                # neg_2 = self.dis(self.my_index_select(source_learn_user, per), source_learn_user_transfer).view(-1)
                # pos_label, neg_label = torch.ones(pos_1.size()).to(self.device), torch.zeros(neg_1.size()).to(self.device)
                
                
                # per = torch.randperm(self.args.target_user_num)[:self.args.user_batch_size].to(self.device)
                # neg_1 = self.dis(self.source2target(source_learn_user_stable), self.my_index_select(target_learn_user, per)).view(-1)
                
                # per = torch.randperm(self.args.source_user_num)[:self.args.user_batch_size].to(self.device)
                # neg_2 = self.dis(self.my_index_select(source_learn_user, per), self.target2source(target_learn_user_stable)).view(-1)
                
                # neg_label = torch.zeros(neg_1.size()).to(self.device)
                # self.critic_loss = source_sim_loss.mean() + target_sim_loss.mean() + self.criterion(neg_1, neg_label) + self.criterion(neg_2, neg_label)
                source_similarity_matrix = self.sim_s[per_stable, :][:, per_stable]
                target_similarity_matrix = self.sim_t[per_stable, :][:, per_stable]
                transfer_loss = self.transfer_sim(source_learn_user_transfer, target_learn_user_transfer, source_similarity_matrix, target_similarity_matrix)
          
                # self.critic_loss = self.args.lambda_loss * transfer_loss + (1-self.args.lambda_loss)*(self.criterion(pos_1, pos_label) + self.criterion(pos_2, pos_label) + self.criterion(neg_1, neg_label) + self.criterion(neg_2, neg_label))
                self.critic_loss = transfer_loss
                source_learn_user_ret = torch.cat((self.target2source(target_learn_user[:self.args.shared_user]), source_learn_user[self.args.shared_user:]),dim=0)
                target_learn_user_ret = torch.cat((self.source2target(source_learn_user[:self.args.shared_user]), target_learn_user[self.args.shared_user:]),dim=0)
        else :
            source_learn_user_ret = torch.cat((self.target2source(target_learn_user[:self.args.source_shared_user]), source_learn_user[self.args.source_shared_user:]), dim=0)
            target_learn_user_ret = torch.cat((self.source2target(source_learn_user[:self.args.target_shared_user]), target_learn_user[self.args.target_shared_user:]), dim=0)

        return source_learn_user_ret, source_learn_item, target_learn_user_ret, target_learn_item
    
    
    