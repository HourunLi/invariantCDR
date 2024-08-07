import torch.optim as optim
import torch
import time
import numpy as np
import os.path as osp
import pandas as pd
from shutil import copyfile
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
from torch_geometric.data import DataLoader, InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data
from torch_geometric.utils import degree
from torch.nn import functional as F
from torch import nn
from tqdm import tqdm
from utils.utils import *
from torch.autograd import Variable
from copy import deepcopy
from itertools import repeat
from model.invariantCDR import invariantCDR
from utils import torch_utils, helper
from datetime import datetime
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold


class Runner(object):
    def __init__(self, args, recmodel, train_batch, source_valid_batch, source_test_batch, target_valid_batch, target_test_batch, source_dev_batch, target_dev_batch, start_epoch=1, writer=None, **kwargs):
        seed_everything(args.seed)
        self.args = args
        self.device = args.device
        self.recmodel = recmodel.to(args.device)
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.optimizer = torch_utils.get_optimizer(args.optim, self.recmodel.parameters(), self.lr, self.weight_decay)
        self.epoch = args.epoch
        self.start_epoch = start_epoch
        self.train_batch = train_batch
        self.source_valid_batch = source_valid_batch
        self.source_test_batch = source_test_batch
        self.target_valid_batch = target_valid_batch
        self.target_test_batch = target_test_batch
        self.source_dev_batch = source_dev_batch
        self.target_dev_batch = target_dev_batch
        self.model_save_dir = args.model_save_dir
        helper.ensure_dir(self.model_save_dir, verbose=True)
        # helper.save_config(vars(args), self.model_save_dir + '/config.json', verbose=True)
        # helper.print_config(vars(args))
        self.file_logger = helper.FileLogger(self.model_save_dir + '/' + args.log, header="# epoch\ttrain_loss\tdev_loss\tdev_score\tbest_dev_score")
        self.criterion = nn.BCEWithLogitsLoss().to(args.device)
        self.max_patience = args.patience

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)
        
    def unpack_batch_predict(self, batch):
        inputs = [Variable(b.to(self.device)) for b in batch]
        user_index = inputs[0]
        item_index = inputs[1]
        return user_index, item_index

    def unpack_batch(self, batch):
        inputs = [Variable(b.to(self.device)) for b in batch]
        source_user = inputs[0]
        source_pos_item = inputs[1]
        source_neg_item = inputs[2]
        target_user = inputs[3]
        target_pos_item = inputs[4]
        target_neg_item = inputs[5]
        return source_user, source_pos_item, source_neg_item, target_user, target_pos_item, target_neg_item
            
    def HingeLoss(self, pos, neg):
        gamma = torch.tensor(self.args.margin)
        gamma = gamma.to(self.device)
        return F.relu(gamma - pos + neg).mean()
    
    def my_index_select_embedding(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = memory(index)
        ans = ans.view(tmp)
        return ans

    def my_index_select(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = torch.index_select(memory, 0, index)
        ans = ans.view(tmp)
        return ans
    
    def reconstruct_graph(self, batch, source_UV, source_VU, target_UV, target_VU):
        self.recmodel.train()
        self.optimizer.zero_grad()

        source_user, source_pos_item, source_neg_item, target_user, target_pos_item, target_neg_item = self.unpack_batch(batch)
        self.source_user, self.source_item, self.target_user, self.target_item = self.recmodel(source_UV, source_VU, target_UV, target_VU)
        
        # print(self.source_user[0:2])
        # print(self.source_item[0:2])
        # print(self.target_user[0:2])
        # print(self.target_item[0:2])
        
        source_user_feature = self.my_index_select(self.source_user, source_user)
        source_item_pos_feature = self.my_index_select(self.source_item, source_pos_item)
        source_item_neg_feature = self.my_index_select(self.source_item, source_neg_item)

        target_user_feature = self.my_index_select(self.target_user, target_user)
        target_item_pos_feature = self.my_index_select(self.target_item, target_pos_item)
        target_item_neg_feature = self.my_index_select(self.target_item, target_neg_item)

        pos_source_score = self.recmodel.source_predict_dot(source_user_feature, source_item_pos_feature)
        neg_source_score = self.recmodel.source_predict_dot(source_user_feature, source_item_neg_feature)
        pos_target_score = self.recmodel.target_predict_dot(target_user_feature, target_item_pos_feature)
        neg_target_score = self.recmodel.target_predict_dot(target_user_feature, target_item_neg_feature)

        source_pos_labels, source_neg_labels = torch.ones(pos_source_score.size()).to(self.device), torch.zeros(neg_source_score.size()).to(self.device)
        target_pos_labels, target_neg_labels = torch.ones(pos_target_score.size()).to(self.device), torch.zeros(neg_target_score.size()).to(self.device)

        recommendation_loss = self.criterion(pos_source_score, source_pos_labels) + self.criterion(neg_source_score, source_neg_labels) + \
            self.criterion(pos_target_score, target_pos_labels) + self.criterion(neg_target_score, target_neg_labels)
        
        loss = self.args.lambda_critic * self.recmodel.critic_loss + (1 - self.args.lambda_critic) * recommendation_loss
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def evaluate_embedding(self, source_UV=None, source_VU=None, target_UV=None, target_VU=None):
        self.source_user, self.source_item, self.target_user, self.target_item = self.recmodel(source_UV, source_VU, target_UV, target_VU)
        return
    
    def source_predict(self, batch):
        user_index, item_index = self.unpack_batch_predict(batch)

        user_feature = self.my_index_select(self.source_user, user_index)
        item_feature = self.my_index_select(self.source_item, item_index)
        user_feature = user_feature.view(user_feature.size()[0], 1, -1)
        user_feature = user_feature.repeat(1, item_feature.size()[1], 1)
        score = self.recmodel.source_predict_dot(user_feature, item_feature)
        return score.view(score.size()[0], score.size()[1])

    def target_predict(self, batch):
        user_index, item_index = self.unpack_batch_predict(batch)

        user_feature = self.my_index_select(self.target_user, user_index)
        item_feature = self.my_index_select(self.target_item, item_index)

        user_feature = user_feature.view(user_feature.size()[0], 1, -1)
        user_feature = user_feature.repeat(1, item_feature.size()[1], 1)

        score = self.recmodel.target_predict_dot(user_feature, item_feature)
        return score.view(score.size()[0], score.size()[1])

    def predict(self, dataloder, choose):
        MRR = 0.0
        NDCG_1 = 0.0
        NDCG_5 = 0.0
        NDCG_10 = 0.0
        HT_1 = 0.0
        HT_5 = 0.0
        HT_10 = 0.0

        valid_entity = 0.0
        for i, batch in enumerate(dataloder):
            if choose:
                predictions = self.source_predict(batch)
            else:
                predictions = self.target_predict(batch)
            # print(predictions)
            for pred in predictions:
                rank = (-pred).argsort().argsort()[0].item()
                # sorted_pred, _ = torch.sort(pred, descending=True)
                # print(sorted_pred[:20])
                valid_entity += 1
                MRR += 1 / (rank + 1)
                if rank < 1:
                    NDCG_1 += 1 / np.log2(rank + 2)
                    HT_1 += 1
                if rank < 5:
                    NDCG_5 += 1 / np.log2(rank + 2)
                    HT_5 += 1
                if rank < 10:
                    NDCG_10 += 1 / np.log2(rank + 2)
                    HT_10 += 1
                if valid_entity % 100 == 0:
                    print('.', end='')

        s_mrr = MRR / valid_entity
        s_ndcg_5 = NDCG_5 / valid_entity
        s_ndcg_10 = NDCG_10 / valid_entity
        s_hr_1 = HT_1 / valid_entity
        s_hr_5 = HT_5 / valid_entity
        s_hr_10 = HT_10 / valid_entity

        return s_mrr, s_ndcg_5, s_ndcg_10, s_hr_1, s_hr_5, s_hr_10
    
    def train(self):
        patience = 0
        s_dev_score_history = [0]
        t_dev_score_history = [0]

        current_lr = self.lr
        if self.start_epoch >= self.args.rectify_epoch:
            self.recmodel.rectify_flag = 1
            # current_lr = self.args.lr_transfer
            
        global_step = 0
        global_start_time = time.time()
        format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/epoch), lr: {:.6f}'
        max_steps = len(self.train_batch) * self.epoch

        best_s_hit = -1
        best_s_ndcg = -1
        best_t_hit = -1
        best_t_ndcg = -1

        # start training
        for epoch in range(self.start_epoch, self.epoch + 1):
            train_loss = 0
            start_time = time.time()
            for i, batch in enumerate(self.train_batch):
                # print(global_step)
                global_step += 1
                loss = self.reconstruct_graph(batch, self.args.source_UV, self.args.source_VU, self.args.target_UV, self.args.target_VU)
                train_loss += loss

            duration = time.time() - start_time
            train_loss = train_loss/len(self.train_batch)
            print(format_str.format(datetime.now(), global_step, max_steps, epoch, self.epoch, train_loss, duration, current_lr))
            if epoch % self.args.log_epoch:
                continue

            # eval recmodel
            print("Evaluating on dev set...")
            self.recmodel.eval()
            self.evaluate_embedding(self.args.source_UV, self.args.source_VU, self.args.target_UV, self.args.target_VU)
            s_mrr, s_ndcg_5, s_ndcg_10, s_hr_1, s_hr_5, s_hr_10 = self.predict(self.source_valid_batch, 1)
            t_mrr, t_ndcg_5, t_ndcg_10, t_hr_1, t_hr_5, t_hr_10 = self.predict(self.target_valid_batch, 0)

            print("\nsource: \t mrr: {:.6f}\t ndcg_5: {:.4f}\t ndcg_10: {:.4f}\t hit@1:{:.6f}\t hit@5:{:.4f}\t hit@10: {:.4f}".format(s_mrr, s_ndcg_5, s_ndcg_10, s_hr_1, s_hr_5, s_hr_10))
            print("target: \t mrr: {:.6f}\t ndcg_5: {:.4f}\t ndcg_10: {:.4f}\t hit@1:{:.6f}\t hit@5:{:.4f}\t hit@10: {:.4f}".format(t_mrr, t_ndcg_5, t_ndcg_10, t_hr_1, t_hr_5, t_hr_10))

            s_dev_score = s_mrr
            t_dev_score = t_mrr

            if s_dev_score > max(s_dev_score_history):
                print("source best!")
                s_mrr, s_ndcg_5, s_ndcg_10, s_hr_1, s_hr_5, s_hr_10 = self.predict(self.source_test_batch, 1)
                print("\nsource: \t mrr: {:.6f}\t ndcg_5: {:.4f}\t ndcg_10: {:.4f}\t hit@1:{:.6f}\t hit@5:{:.4f}\t hit@10: {:.4f}".format(s_mrr, s_ndcg_5, s_ndcg_10, s_hr_1, s_hr_5, s_hr_10))

            if t_dev_score > max(t_dev_score_history):
                print("target best!")
                t_mrr, t_ndcg_5, t_ndcg_10, t_hr_1, t_hr_5, t_hr_10 = self.predict(self.target_test_batch, 0)
                print("target: \t mrr: {:.6f}\t ndcg_5: {:.4f}\t ndcg_10: {:.4f}\t hit@1:{:.6f}\t hit@5:{:.4f}\t hit@10: {:.4f}".format(t_mrr, t_ndcg_5, t_ndcg_10, t_hr_1, t_hr_5, t_hr_10))

            self.file_logger.log(
                "{}\t{:.6f}\t{:.4f}\t{:.4f}".format(epoch, train_loss, s_dev_score, max([s_dev_score] + s_dev_score_history)))

            print(
                "epoch {}: train_loss = {:.6f}, source_hit@10 = {:.4f}, source_ndcg@10 = {:.4f}, target_hit@10 = {:.4f}, target_ndcg@10 = {:.4f}".format(
                        epoch, \
                    train_loss, s_hr_10, s_ndcg_10, t_hr_10, t_ndcg_10))

            # save
            model_file = self.model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
            if epoch > 1 and (s_dev_score > max(s_dev_score_history) or t_dev_score > max(t_dev_score_history)):
                patience = 0
                torch.save({
                    'epoch':epoch,
                    'model_state_dict': self.recmodel.state_dict(),
                }, model_file)
                copyfile(model_file, self.model_save_dir + '/best_model.pt')
                print("new best model saved.")
            else:
                patience += 1
                if epoch > self.args.min_epoch and patience > self.max_patience:
                    print("early termination of training")
                    return
                
            if epoch >= self.args.rectify_epoch and self.recmodel.rectify_flag == 0:
                self.recmodel.rectify_flag = 1
                print("shift to transfer learning mode")
                # model_file = self.model_save_dir + '/similarity.pt'.format(epoch)
                # torch.save({
                #     'epoch':epoch,
                #     'model_state_dict': self.recmodel.state_dict(),
                # }, model_file)
                # current_lr = self.args.lr_transfer
                
            # lr schedule
            if epoch > self.args.decay_epoch and s_dev_score + t_dev_score < s_dev_score_history[-1] + t_dev_score_history[-1] and self.args.optim in ['sgd', 'adagrad', 'adadelta', 'adam']:
                current_lr *= self.args.lr_decay
                self.update_lr(current_lr)

            s_dev_score_history += [s_dev_score]
            t_dev_score_history += [t_dev_score]
            print("")   

        return
    
    def eval(self, epoch, data, dev_score_history):
        self.recmodel.eval()
        decay_switch = 0

        node_emb_shared, node_emb_specific = self.recmodel.get_node_embedding(data, reshape = True)
        
        valid_start = time.time()
        for idx, cur_domain in enumerate(self.valid_dataloader):
            metrics = self.recmodel.predict(idx, self.valid_dataloader[cur_domain], node_emb_shared, node_emb_specific, data, data['user_base'], data['item_base'])
            print("\n-------------------" + cur_domain + "--------------------")
            print(metrics)

            if metrics["NDCG_10"] > max(dev_score_history[idx]):
                print(f"{cur_domain} get better results! Predict on test dataset")
                if self.args.save:
                    self.recmodel.save()
                    print("best model saved!")
                # If the NDCG score on the validation set for a domain is higher than 
                # the best score seen so far (max(dev_score_history[idx])), 
                # the model is tested on the test set, and the results are printed.
                test_metrics = self.recmodel.predict(idx, self.test_dataloader[cur_domain], node_emb_shared, node_emb_specific, data, data['user_base'], data['item_base'])
                print(test_metrics)
            else:
                decay_switch += 1
            dev_score_history[idx].append(metrics["NDCG_10"])
            
        print("valid time:  ", (time.time() - valid_start), "(s)")
        return decay_switch
        