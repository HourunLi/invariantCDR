import argparse
import torch
import os

parser = argparse.ArgumentParser(description='invariantCDR Arguments.')

# model
parser.add_argument('--domains', type=str, default="sport_cloth || electronic_cell, sport_cloth || game_video, uk_de_fr_ca_us", help='specify none ("none") or a few source markets ("-" seperated) to augment the data for training')
parser.add_argument('--feature_dim', type=int, default=192, help='Initialize network embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=192, help='hidden dimensions')
parser.add_argument("--num_latent_factors", type=int, default=8, help="latent factors")
parser.add_argument('--conv_layers', type=int, default=4, help="number of graph convolutional layers")
parser.add_argument('--proj_layers', type=int, default=1, help="number of graph projection layers")
parser.add_argument('--mask_rate', type=float, default=0.1, help='mask rate of interactions')
parser.add_argument('--weight_decay', type=float, default=2e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--A_split', action='store_true', help='split the model')
parser.add_argument('--a_fold', type=int, default=100, help="the fold num used to split large adj matrix, like gowalla")
parser.add_argument('--JK', type=str, default='sum', choices=['last', 'sum'])
parser.add_argument('--residual', type=int, default=1, choices=[0, 1])
parser.add_argument('--projection', type=int, default=1, choices=[0, 1])
parser.add_argument('--num_negative', type=int, default=10, help='num of negative samples during training')
parser.add_argument('--maxlen', type=int, default=10, help='num of item sequence')
parser.add_argument('--static_sample', action='store_true', help='accelerate the dataloader')
parser.add_argument('--test_sample_number', type=int, default=999)

# train
parser.add_argument('--epoch', type=int, default=200, help='number of epoches')
parser.add_argument("--min_epoch", type=int, default=50, help="min epoch")
parser.add_argument("--patience", type=int, default=50, help="patience for early stop")
parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
parser.add_argument('--user_batch_size', type=int, default=1024, help='Training batch size.')
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam',help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--lr_transfer', type=float, default=2e-4, help='learning rate')
parser.add_argument('--lr_decay', type=float, default=0.95, help='Learning rate decay rate.')

# config
parser.add_argument("--mode", type=str, default="train", help="train, eval")
parser.add_argument('--cuda', action='store_true', help='use of cuda')
parser.add_argument("--device", type=str, default="gpu", help="training device")
parser.add_argument("--device_id", type=str, default="0", help="device id for gpu")
parser.add_argument('--seed', type=int, default=32, help='manual seed init')
parser.add_argument('--transfer_epoch', type=int, default=50, help='transfer learning after this epoch.')
parser.add_argument('--decay_epoch', type=int, default=5, help='Decay learning rate after this epoch.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')
parser.add_argument('--load', dest='load', action='store_true', default=False,  help='Load pretrained model.')
parser.add_argument('--save', action='store_true', help='save model?')
parser.add_argument('--model_name', type=str, default='default', help='Model name under which to save models.')


#hyper parameters
parser.add_argument('--beta', type=float, default=1.5)
parser.add_argument('--leakey', type=float, default=0.1)
parser.add_argument('--lambda_loss', type=float, default=1, help='the parameter of loss function')
parser.add_argument('--dropout', type=float, default=0.2, help='random drop out rate')
parser.add_argument('--keep_prob', type=float,default=0.6, help="the batch size for bpr loss training procedure")
parser.add_argument('--tau', type=float, default=1, help="temperature parameter for scaling the similarity scores")
parser.add_argument('--sim_threshold', type=float, default=0.7, help="similarity threshold")


args = parser.parse_args()

# set the running device
if int(args.device_id) >= 0 and torch.cuda.is_available():
    args.device = torch.device("cuda:{}".format(args.device_id))
    print("using gpu:{} to train the model".format(args.device_id))
else:
    args.device = torch.device("cpu")
    print("using cpu to train the model")