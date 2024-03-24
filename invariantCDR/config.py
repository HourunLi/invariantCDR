import argparse
import torch
import os

parser = argparse.ArgumentParser(description='invariantCDR Arguments.')

# data
parser.add_argument('--domains', type=str, default="sport_cloth || electronic_cell, sport_cloth || game_video, uk_de_fr_ca_us", help='specify none ("none") or a few source markets ("-" seperated) to augment the data for training')
parser.add_argument('--task', type=str, default='dual-user-intra', help='dual-user-intra, dual-user-inter, multi-item-intra, multi-user-intra')
parser.add_argument("--nfeat", type=int, default=128, help="dim of input feature")

# model
parser.add_argument('--aggregator', type=str, default='mean', help='switching the user-item aggregation')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam',
                    help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument("--num_latent_factors", type=int, default=8, help="latent factors")
parser.add_argument('--num_layers', type=int, default=4, help="number of graph layers")
parser.add_argument('--head_layers', type=int, default=1, help="number of graph head layers")
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--l2_reg', type=float, default=1e-7, help='the L2 weight')
parser.add_argument('--lr_decay', type=float, default=0.98, help='decay learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='decay learning rate')
parser.add_argument('--node_feature', type=int, default=128, help='user or item embedding dimensions')
# parser.add_argument('--item_dim', type=int, default=128, help='item embedding dimensions')
parser.add_argument('--aug', type=str, default='random4', help='augmentation methods')
parser.add_argument('--latent_dim', type=int, default=128, help='latent dimensions')
parser.add_argument('--drop_ratio', type=float, default=0.3, delp = "drop rtatio")
parser.add_argument('--num_negative', type=int, default=10, help='num of negative samples during training')
parser.add_argument('--maxlen', type=int, default=10, help='num of item sequence')
parser.add_argument('--dropout', type=float, default=0.3, help='random drop out rate')
parser.add_argument('--save', action='store_true', help='save model?')
parser.add_argument('--lambda', type=float, default=50, help='the parameter of EASE')
parser.add_argument('--lambda_a', type=float, default=0.5, help='for our aggregators')
parser.add_argument('--lambda_loss', type=float, default=0.4, help='the parameter of loss function')
parser.add_argument('--static_sample', action='store_true', help='accelerate the dataloader')
# others
parser.add_argument('--epoch', type=int, default=100, help='number of epoches')
parser.add_argument("--min_epoch", type=int, default=50, help="min epoch")
parser.add_argument("--mode", type=str, default="train", help="train, eval")
parser.add_argument('--cuda', action='store_true', help='use of cuda')
parser.add_argument("--device", type=str, default="gpu", help="training device")
parser.add_argument("--device_id", type=str, default="0", help="device id for gpu")
parser.add_argument('--seed', type=int, default=42, help='manual seed init')
parser.add_argument("--log_interval", type=int, default=10, help="every n epoches to log")
parser.add_argument('--decay_epoch', type=int, default=10, help='Decay learning rate after this epoch.')

args = parser.parse_args()

# set the running device
if int(args.device_id) >= 0 and torch.cuda.is_available():
    args.device = torch.device("cuda:{}".format(args.device_id))
    print("using gpu:{} to train the model".format(args.device_id))
else:
    args.device = torch.device("cpu")
    print("using cpu to train the model")