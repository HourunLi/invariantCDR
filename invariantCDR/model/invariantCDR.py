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
import copy

class invariantCDR(nn.Module):
    def __init__(self, args, device):
        super(invariantCDR, self).__init__()
        self.args = args
        self.device = device
        self.momentum = args.momentum
        self.source_disenEncoder_online = disenEncoder(args)
        self.target_disenEncoder_online = disenEncoder(args)
        
        self.source_disenEncoder_goal = copy.deepcopy(self.source_disenEncoder_online)
        self.target_disenEncoder_goal = copy.deepcopy(self.target_disenEncoder_online)
        
        # initialize the target and online encoder for both source and target
        for param_q, param_k in zip(self.source_disenEncoder_online.parameters(), self.source_disenEncoder_goal.parameters()):
            param_k.data.copy_(param_q.data)  
            param_k.requires_grad = False  
            
        for param_q, param_k in zip(self.target_disenEncoder_online.parameters(), self.target_disenEncoder_goal.parameters()):
            param_k.data.copy_(param_q.data) 
            param_k.requires_grad = False
              
        self.s2t_transfer = FactorDomainTransformer(args)
        self.t2s_transfer = FactorDomainTransformer(args)
        
        self.k_tau = args.k_tau
        self.batch_tau = args.batch_tau 
        self.similarity_tau = args.similarity_tau
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
        self.rectify_flag = 0
        self.K = args.num_latent_factors 
        self.d = args.feature_dim // self.K
        
        self.source_user_center = torch.rand((self.K, self.d), requires_grad=True).to(self.device)
        self.target_user_center = torch.rand((self.K, self.d), requires_grad=True).to(self.device)
        
        self.dropout = self.args.dropout
    #     self.__init_weight()
        
    # def __init_weight(self):
    #     nn.init.normal_(self.source_user_embedding.weight, std = 0.1)
    #     nn.init.normal_(self.target_user_embedding.weight, std = 0.1)     
    #     nn.init.normal_(self.source_item_embedding.weight, std = 0.1)
    #     nn.init.normal_(self.target_item_embedding.weight, std = 0.1)

    @torch.no_grad()
    def _update_target_branch(self, momentum):
        for param_o, param_t in zip(self.source_disenEncoder_online.parameters(), self.source_disenEncoder_goal.parameters()):
            param_t.data = param_t.data * momentum + param_o.data * (1 - momentum)
            
        for param_o, param_t in zip(self.target_disenEncoder_online.parameters(), self.target_disenEncoder_goal.parameters()):
            param_t.data = param_t.data * momentum + param_o.data * (1 - momentum)
            
   
    def source_predict_nn(self, user_embedding, item_embedding):
        fea = torch.cat((user_embedding, item_embedding), dim=-1)
        return self.predict_source(fea)

    def target_predict_nn(self, user_embedding, item_embedding):
        fea = torch.cat((user_embedding, item_embedding), dim=-1)
        return self.predict_target(fea)

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
    
    def HingeLoss(self, pos, neg):
        pos = F.sigmoid(pos)
        neg = F.sigmoid(neg)
        gamma = torch.tensor(self.args.margin)
        if self.args.cuda:
            gamma = gamma.cuda()
        return F.relu(gamma - pos + neg).mean()

    # def min_max_norm(self, embeddings):
    #     min_vals = embeddings.min(dim=1, keepdim=True)[0]
    #     max_vals = embeddings.max(dim=1, keepdim=True)[0]
    #     embeddings = (embeddings - min_vals) / (max_vals - min_vals)
    #     return embeddings
    
    # def row_wise_norm(self, embeddings):
    #     row_sums = embeddings.sum(dim=1, keepdim=True)
    #     embeddings = embeddings / row_sums
    #     return embeddings

    # def _cal_kernel_affinity(self, embeddings, step: int = 5):
    #     norm_embeddings = F.normalize(embeddings, p=2, dim=-1)
    #     B, K, d = norm_embeddings.size()
    #     norm_embeddings = torch.reshape(norm_embeddings, (B, K*d))
    #     G = (2 * K - 2 * (norm_embeddings @ norm_embeddings.t())).clamp(min=0.)
    #     G = torch.exp(-G / (self.similarity_tau * K))
    #     G = G / G.sum(dim=1, keepdim=True)
    #     G = torch.matrix_power(G, step)
    #     # print(G[:2])
    #     G = torch.eye(B).to(self.device) * self.args.alpha + G * (1 - self.args.alpha)
    #     return G
    
    def _cal_kernel_affinity(self, embeddings, step: int = 5):
        norm_embeddings = F.normalize(embeddings, p=2, dim=-1)
        flag = len(norm_embeddings.size()) > 2
        if flag:
            B, K, d = norm_embeddings.size()
            # (K, B, D)
            norm_embeddings = norm_embeddings.permute(1, 0, 2).contiguous()
            similarity =  torch.einsum('kid,kjd->kij', norm_embeddings, norm_embeddings)
        else:
            B, d = norm_embeddings.size()
            similarity = torch.einsum('id,jd->ij', norm_embeddings, norm_embeddings)

        G = (2 - 2 * similarity).clamp(min=0.)
        G = torch.exp(-G / self.similarity_tau)
        G = G / G.sum(dim=-1, keepdim=True)
        G = torch.matrix_power(G, step)
        identity_matrix = torch.eye(B).to(self.device).unsqueeze(0) if flag else torch.eye(B).to(self.device)
        G = identity_matrix * self.args.alpha + G * (1 - self.args.alpha)
        return G
    
    def cal_similarity_matrix(self, source_user, target_user):
        source_sim = self._cal_kernel_affinity(source_user).detach()
        target_sim = self._cal_kernel_affinity(target_user).detach()
        return source_sim, target_sim
    
    def inter_cl(self, x_q, x_k, center_v, mask_pos=None):
        B, K, d = x_q.size()
        if mask_pos is None:
            mask_pos = torch.eye(B).to(self.device)
        assert len(mask_pos.size()) == 2

        # print(mask_pos[:, :3, :5])
        ck = F.normalize(center_v, dim=-1)
        p_k_x_ = torch.einsum('bkd,kd->bk', F.normalize(x_q, dim=-1), ck)
        
        #（B, K, 1）
        p_k_x = F.softmax(p_k_x_ / self.k_tau, dim=-1) # equation 4
        p_k_x = p_k_x.unsqueeze(-1)

        # torch.reshape(mask_pos, (B, K, B))
        # mask_pos = mask_pos * p_k_x
        # mask_pos = mask_pos.sum(1)
        # assert len(mask_pos.size()) == 2
        
        # (K, B, d)
        x_q_abs = x_q.norm(dim=-1)
        x_k_abs = x_k.norm(dim=-1)
        x_q = x_q.permute(1, 0, 2).contiguous()
        x_k = x_k.permute(1, 0, 2).contiguous()
        x_q_abs = x_q_abs.permute(1, 0).contiguous()
        x_k_abs = x_k_abs.permute(1, 0).contiguous()
        # (K, B, B)
        sim_matrix = torch.einsum('kid,kjd->kij', x_q, x_k) / (1e-8 + torch.einsum('ki,kj->kij', x_q_abs, x_k_abs))
        # (B ,K, B)
        sim_matrix = sim_matrix.permute(1, 0, 2).contiguous()
        p_y_xk = F.softmax(sim_matrix / self.batch_tau, dim=-1) # equation 8 in paper

        # (B, K, B)
        q_k = p_k_x * p_y_xk
        # (B, B, K)
        q_k = q_k.permute(0, 2, 1).contiguous()
        # (B, B, K)
        q_k_xy = F.normalize(q_k, dim=-1, p=1) # equation 9
        elbo = q_k_xy * (torch.log(q_k) - torch.log(q_k_xy))
        elbo = elbo.sum(dim = -1)
        # elbo = q_k * (torch.log(p_k_x) + torch.log(p_y_xk) - torch.log(q_k))
        # elbo = elbo.mean(dim = 0)
        # print(f"elbo is {- elbo.mean()}")
        nll_loss = elbo * mask_pos
        loss = -nll_loss.sum() / B
        return loss
    
        # B, K, d = x_q.size()
        # ck = F.normalize(center_v)
        # p_k_x_ = torch.einsum('bkd,kd->bk', F.normalize(x_q, dim=-1), ck)
        # p_k_x = F.softmax(p_k_x_ / self.inter_tau, dim=-1) # equation 4        
        
        # x_q_abs = x_q.norm(dim=-1)
        # x_k_abs = x_k.norm(dim=-1)
        # x_q = torch.reshape(x_q, (B * K, d))
        # x_k = torch.reshape(x_k, (B * K, d))
        # x_q_abs = torch.squeeze(torch.reshape(x_q_abs, (B * K, 1)), 1)
        # x_k_abs = torch.squeeze(torch.reshape(x_k_abs, (B * K, 1)), 1)
        # sim_matrix = torch.einsum('ik,jk->ij', x_q, x_k) / (1e-8 + torch.einsum('i,j->ij', x_q_abs, x_k_abs))
        # sim_matrix = torch.exp(sim_matrix / self.inter_tau)
        # pos_sim = sim_matrix[range(B * K), range(B * K)]
        # score = pos_sim / (sim_matrix.sum(dim=-1) - pos_sim) 
        # p_y_xk = score.view(B, K)# equation 8 in paper
        
        # q_k = torch.einsum('bk,bk->bk', p_k_x, p_y_xk)
        # q_k = F.normalize(q_k, dim=-1)
        # elbo = q_k * (torch.log(p_k_x) + torch.log(p_y_xk) - torch.log(q_k))
        # loss = - elbo.view(-1).mean()
        # return loss

    def intra_cl(self, x_q, x_k, mask_pos=None):
        # uniformed intra contrastive
        # B, K, d = x_q.size()
        # if sim is None:
        #     sim = torch.eye(B).to(self.device)
        # mask_pos = sim.unsqueeze(0)
        # x_q_abs = x_q.norm(dim=-1)
        # x_k_abs = x_k.norm(dim=-1)
        
        # x_q = torch.reshape(x_q, (B * K, d))
        # x_k = torch.reshape(x_k, (B * K, d))
        # x_q_abs = torch.squeeze(torch.reshape(x_q_abs, (B * K, 1)), 1)
        # x_k_abs = torch.squeeze(torch.reshape(x_k_abs, (B * K, 1)), 1)
        
        # #(B*K, B*K)
        # sim_matrix = torch.einsum('ik,jk->ij', x_q, x_k) / (1e-8 + torch.einsum('i,j->ij', x_q_abs, x_k_abs))
        # sim_matrix = F.softmax(sim_matrix / self.intra_tau, dim = -1)
        # # (K, B, B)
        # result = torch.zeros((K, B, B)).to(self.device)
        # for i in range(K):
        #     result[i] = sim_matrix[range(i, B * K, K), range(i, B * K, K)]
        
        # nll_loss = -torch.log(result) * mask_pos / mask_pos.sum(dim=-1, keepdim=True)
        # loss = nll_loss.mean()
        # return loss

        # bi-contrastive
        B, K, d = x_q.size()
        if mask_pos is None:
            mask_pos = torch.eye(B).to(self.device).unsqueeze(0)
        assert len(mask_pos.size()) == 3
        # print(mask_pos[:, :3, :5])
        x_q_abs = x_q.norm(dim=-1)
        x_k_abs = x_k.norm(dim=-1)
        # part 1
        x_q_1 = x_q
        x_k_1 = x_k
        x_q_1_abs = x_q_abs
        x_k_1_abs = x_k_abs
        k_sim = torch.einsum('bid,bjd->bij', x_q_1, x_k_1) / (1e-8 + torch.einsum('bi,bj->bij', x_q_1_abs, x_k_1_abs))
        # k_sim = F.softmax(k_sim / self.k_tau, dim=-1) 
        k_sim = F.softmax(k_sim, dim=-1) 
        score = k_sim[:, range(K), range(K)]
        score = score.view(B, K)
        loss_1 = -torch.log(score).mean()
        # sim_matrix = torch.exp(sim_matrix / self.intra_tau)
        # pos_sim = sim_matrix[:, range(K), range(K)]
        # score = pos_sim / (sim_matrix.sum(dim=-1) - pos_sim) 
        # loss_1 = -torch.log(score).mean()
        
        # part 2
        # (K, B, B)
        x_q_2 = x_q.permute(1, 0, 2).contiguous()
        x_k_2 = x_k.permute(1, 0, 2).contiguous()
        x_q_2_abs = x_q_abs.permute(1, 0).contiguous()
        x_k_2_abs = x_k_abs.permute(1, 0).contiguous()
        # x_q_2_abs = torch.squeeze(torch.reshape(x_q_abs, (K, B, 1)), 2)
        # x_k_2_abs = torch.squeeze(torch.reshape(x_k_abs, (K, B, 1)), 2)
        b_sim = torch.einsum('kid,kjd->kij', x_q_2, x_k_2) / (1e-8 + torch.einsum('ki,kj->kij', x_q_2_abs, x_k_2_abs))
        # b_sim = F.softmax(b_sim / self.batch_tau, dim = -1)
        b_sim = F.softmax(b_sim, dim = -1)
        nll_loss = torch.log(b_sim) * mask_pos
        loss_2 = - nll_loss.sum() / (B * K)
        # print(f"intra loss: loss_1: {loss_1}, loss_2: {loss_2}")
        return (loss_1 + loss_2)/2
    
    # def forward(self, source_UV, source_VU, target_UV, target_VU):
    #     self._update_target_branch(self.momentum)
    #     source_user = self.source_user_embedding(self.source_user_index)
    #     target_user = self.target_user_embedding(self.target_user_index)
    #     source_item = self.source_item_embedding(self.source_item_index)
    #     target_item = self.target_item_embedding(self.target_item_index)

    #     source_user_online, source_item_online = self.source_disenEncoder_online(source_user, source_item, source_UV, source_VU)
    #     target_user_online, target_item_online = self.target_disenEncoder_online(target_user, target_item, target_UV, target_VU)
    #     source_user_goal, source_item_goal = self.source_disenEncoder_goal(source_user, source_item, source_UV, source_VU)
    #     target_user_goal, target_item_goal = self.target_disenEncoder_goal(target_user, target_item, target_UV, target_VU)
    #     source_user_concat, target_user_concat = None, None

    #     if self.training:
    #         per_stable = torch.randperm(self.args.shared_user)[:self.args.user_batch_size].to(self.device)
    #         per_random_source = torch.randperm(self.args.source_user_num)[:self.args.user_batch_size].to(self.device)
    #         per_random_target = torch.randperm(self.args.target_user_num)[:self.args.user_batch_size].to(self.device)
            
    #         sim_random_s, sim_random_t, sim_shared_s, sim_shared_t = None, None, None, None
    #         if self.rectify_flag:
    #             sim_random_s, sim_random_t = self.cal_similarity_matrix(source_user_goal[per_random_source], target_user_goal[per_random_target])
    #             sim_shared_s, sim_shared_t = self.cal_similarity_matrix(source_user_goal[per_stable], target_user_goal[per_stable])
    #         mp = [sim_random_s, sim_random_t, sim_shared_s, sim_shared_t]
            
    #         l_inter = (self.inter_cl(self.s2t_transfer.forward_user(source_user_online[per_stable]), target_user_goal[per_stable], self.target_user_center, mp[3]) + 
    #                    self.inter_cl(self.t2s_transfer.forward_user(target_user_online[per_stable]), source_user_goal[per_stable], self.source_user_center, mp[2])) / 2            
    #         l_intra = (self.intra_cl(source_user_online[per_random_source], source_user_goal[per_random_source], mp[0]) + 
    #                    self.intra_cl(target_user_online[per_random_target], target_user_goal[per_random_target], mp[1])) / 2            
    #         self.critic_loss = self.args.beta_inter * l_inter + (1-self.args.beta_inter) * l_intra
            
    #         print(f"inter loss {l_inter}, intra loss: {l_intra}, critic loss: {self.critic_loss}")
    #         source_user_concat = torch.cat((self.t2s_transfer.forward_user(target_user_online[:self.args.shared_user]), source_user_online[self.args.shared_user:]),dim=0)
    #         target_user_concat = torch.cat((self.s2t_transfer.forward_user(source_user_online[:self.args.shared_user]), target_user_online[self.args.shared_user:]),dim=0)
    #     else:
    #         source_user_concat = torch.cat((self.t2s_transfer.forward_user(target_user_online[:self.args.source_shared_user]), source_user_online[self.args.source_shared_user:]), dim=0)
    #         target_user_concat = torch.cat((self.s2t_transfer.forward_user(source_user_online[:self.args.target_shared_user]), target_user_online[self.args.target_shared_user:]), dim=0)

    #     return source_user_concat, source_item_online, target_user_concat, target_item_online

    def forward(self, source_UV, source_VU, target_UV, target_VU):
        self._update_target_branch(self.momentum)
        source_user = self.source_user_embedding(self.source_user_index)
        target_user = self.target_user_embedding(self.target_user_index)
        source_item = self.source_item_embedding(self.source_item_index)
        target_item = self.target_item_embedding(self.target_item_index)

        source_user_origin_online, source_item_origin_online, source_user_disen_online, source_item_disen_online = self.source_disenEncoder_online(source_user, source_item, source_UV, source_VU)
        target_user_origin_online, target_item_origin_online, target_user_disen_online, target_item_disen_online = self.target_disenEncoder_online(target_user, target_item, target_UV, target_VU)
        source_user_origin_goal, source_item_origin_goal, source_user_disen_goal, source_item_disen_goal = self.source_disenEncoder_goal(source_user, source_item, source_UV, source_VU)
        target_user_origin_goal, target_item_origin_goal, target_user_disen_goal, target_item_disen_goal = self.target_disenEncoder_goal(target_user, target_item, target_UV, target_VU)
        source_user_concat, target_user_concat = None, None

        if self.training:
            per_stable = torch.randperm(self.args.shared_user)[:self.args.user_batch_size].to(self.device)
            per_random_source = torch.randperm(self.args.source_user_num)[:self.args.user_batch_size].to(self.device)
            per_random_target = torch.randperm(self.args.target_user_num)[:self.args.user_batch_size].to(self.device)
            
            sim_random_s, sim_random_t, sim_shared_s, sim_shared_t = None, None, None, None
            if self.rectify_flag:
                # for intra
                sim_random_s, sim_random_t = self.cal_similarity_matrix(source_user_disen_goal[per_random_source], target_user_disen_goal[per_random_target])
                # for inter
                sim_shared_s, sim_shared_t = self.cal_similarity_matrix(source_user_origin_goal[per_stable], target_user_origin_goal[per_stable])
            mp = [sim_random_s, sim_random_t, sim_shared_s, sim_shared_t]
            
            l_intra = (self.intra_cl(source_user_disen_online[per_random_source], source_user_disen_goal[per_random_source], mp[0]) + 
                       self.intra_cl(target_user_disen_online[per_random_target], target_user_disen_goal[per_random_target], mp[1])) / 2       
            l_inter = (self.inter_cl(self.t2s_transfer.forward_user(target_user_disen_online[per_stable]), source_user_disen_goal[per_stable], self.source_user_center, mp[2]) +
                       self.inter_cl(self.s2t_transfer.forward_user(source_user_disen_online[per_stable]), target_user_disen_goal[per_stable], self.target_user_center, mp[3])) / 2            
            self.critic_loss = self.args.beta_inter * l_inter + (1-self.args.beta_inter) * l_intra
            
            # print(f"inter loss {l_inter}, intra loss: {l_intra}, critic loss: {self.critic_loss}")
            source_user_concat = torch.cat((self.t2s_transfer.forward_user(target_user_disen_online[:self.args.shared_user]), source_user_disen_online[self.args.shared_user:]),dim=0)
            target_user_concat = torch.cat((self.s2t_transfer.forward_user(source_user_disen_online[:self.args.shared_user]), target_user_disen_online[self.args.shared_user:]),dim=0)
        else:
            source_user_concat = torch.cat((self.t2s_transfer.forward_user(target_user_disen_online[:self.args.source_shared_user]), source_user_disen_online[self.args.source_shared_user:]), dim=0)
            target_user_concat = torch.cat((self.s2t_transfer.forward_user(source_user_disen_online[:self.args.target_shared_user]), target_user_disen_online[self.args.target_shared_user:]), dim=0)

        return source_user_concat, source_item_disen_online, target_user_concat, target_item_disen_online
    