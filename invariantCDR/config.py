import argparse
import torch
import os

parser = argparse.ArgumentParser(description='invariantCDR Arguments.')

# model
parser.add_argument('--domains', type=str, default="sport_cloth || electronic_cell, sport_cloth || game_video, uk_de_fr_ca_us", help='specify none ("none") or a few source markets ("-" seperated) to augment the data for training')
parser.add_argument('--feature_dim', type=int, default=126, help='Initialize network embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=126, help='hidden dimensions')
parser.add_argument("--num_latent_factors", type=int, default=3, help="latent factors")
parser.add_argument('--conv_layers', type=int, default=4, help="number of graph convolutional layers")
parser.add_argument('--proj_layers', type=int, default=1, help="number of graph projection layers")
parser.add_argument('--weight_decay', type=float, default=2e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--A_split', action='store_true', help='split the model')
parser.add_argument('--a_fold', type=int, default=100, help="the fold num used to split large adj matrix, like gowalla")
parser.add_argument('--JK', type=str, default='sum', choices=['last', 'sum'])
parser.add_argument('--residual', type=int, default=1, choices=[0, 1])
parser.add_argument('--projection', type=int, default=1, choices=[0, 1])
parser.add_argument('--num_negative', type=int, default=100, help='num of negative samples during training')
parser.add_argument('--test_sample_number', type=int, default=999)

# train
parser.add_argument('--epoch', type=int, default=80, help='number of epoches')
parser.add_argument("--min_epoch", type=int, default=50, help="min epoch")
parser.add_argument('--transfer_epoch', type=int, default=40, help='transfer learning after this epoch.')
parser.add_argument('--log_epoch', type=int, default=5, help='log every log_epoch')
parser.add_argument("--patience", type=int, default=30, help="patience for early stop")
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--user_batch_size', type=int, default=256, help='Training user batch size.')
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam',help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--lr', type=float, default=4e-3, help='learning rate')
parser.add_argument('--lr_decay', type=float, default=0.95, help='Learning rate decay rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='exponential moving average momentum')

# config
parser.add_argument("--mode", type=str, default="train", help="train, eval")
parser.add_argument('--cuda', action='store_true', help='use of cuda')
parser.add_argument("--device", type=str, default="gpu", help="training device")
parser.add_argument("--device_id", type=str, default="0", help="device id for gpu")
parser.add_argument('--seed', type=int, default=32, help='manual seed init')
parser.add_argument('--decay_epoch', type=int, default=10, help='Decay learning rate after this epoch.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')
parser.add_argument('--load', dest='load', action='store_true', default=False,  help='Load pretrained model.')
parser.add_argument('--aggregate', action='store_true', default=False,  help='aggregate item features of similar users.')
parser.add_argument('--save', action='store_true', help='save model?')
parser.add_argument('--model_name', type=str, default='default', help='Model name under which to save models.')

#hyper parameters
parser.add_argument('--leakey', type=float, default=0.05)
parser.add_argument('--alpha', type=float, default=0.5, help='the weight of I in similarity')
parser.add_argument('--lambda_critic', type=float, default=0.4, help='the parameter of critic loss function')
parser.add_argument('--beta_inter', type=float, default=0.4)
parser.add_argument('--dropout', type=float, default=0.5, help='random drop out rate')
parser.add_argument('--inter_tau', type=float, default=0.2, help="temperature parameter for scaling the inter similarity scores")
parser.add_argument('--intra_tau', type=float, default=1, help="temperature parameter for scaling the intra similarity scores")
parser.add_argument('--similarity_tau', type=float, default=0.1, help="temperature parameter for scaling the intra similarity scores")
args = parser.parse_args()

# set the running device
if int(args.device_id) >= 0 and torch.cuda.is_available():
    args.device = torch.device("cuda:{}".format(args.device_id))
    print("using gpu:{} to train the model".format(args.device_id))
else:
    args.device = torch.device("cpu")
    print("using cpu to train the model")