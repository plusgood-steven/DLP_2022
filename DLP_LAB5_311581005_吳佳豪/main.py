#%%
import argparse
import os
import random

import torch
from torch.utils.data import DataLoader

import logging

from dataset import Iclver_dataset,get_json_labels
from model import _netD, _netG, dc_netD, dc_netG
from train import ACGan_train,DCGan_train


#%%
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default="DCGan", type=str, help='choice of model name ACGan or DCGan')
    parser.add_argument('--num_classes', default=24, type=int, help='classes number')
    parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0, type=float, help='weight decay')
    parser.add_argument('--beta', default=0.5, type=float, help='momentum term for adam')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--log_dir', default='./log/train', help='base directory to save logs')
    parser.add_argument('--model_dir', default='', help='base directory to save logs')
    parser.add_argument('--test_json_name', default='test', help='test json file name')
    parser.add_argument('--dataset_root', default='./dataset', help='root directory for data')
    parser.add_argument('--generator_iter_loop',type=int, default=5, help='every iter generator train times')
    parser.add_argument('--label_smoothing', default=0, type=float, help='label smoothing ratio')
    parser.add_argument('--realfake_loss_ratio', default=0.5, type=float, help='realfake loss ratio(p)')
    parser.add_argument('--classes_loss_ratio', default=0.5, type=float, help='classes loss ratio')
    parser.add_argument('--realfake_loss_decay', default=0, type=float, help='realfake loss decay and classes loss increasing')
    parser.add_argument('--realfake_start_epoch', default=0, type=float, help='realfake loss  over start epochs')
    parser.add_argument('--epochs', type=int, default=1000, help='epoch size')
    parser.add_argument('--seed', default=123, type=int, help='manual seed')
    parser.add_argument('--z_dim', type=int, default=64, help='size of the latent z vector')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading threads')
    parser.add_argument('--cuda', default=True, action='store_true')
    parser.add_argument('--cuda_num', type=int, default=0)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = f'cuda:{args.cuda_num}'
        torch.backends.cudnn.benchmark = True
    else:
        device = 'cpu'

    print("device:",device)

    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)    
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs('%s/img/' % args.log_dir, exist_ok=True)

    if os.path.exists('./{}/train_record.log'.format(args.log_dir)):
        os.remove('./{}/train_record.log'.format(args.log_dir))

    logging.basicConfig(filename='./{}/train_record.log'.format(args.log_dir), encoding='utf-8', level=logging.INFO)
    logging.info('args: {}\n'.format(args))

    train_dataset = Iclver_dataset(f"{args.dataset_root}/data",f"{args.dataset_root}/train.json",f"{args.dataset_root}/objects.json")
    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,pin_memory=True,drop_last=True)

    test_condition_labels = get_json_labels(f"{args.dataset_root}/{args.test_json_name}.json",f"{args.dataset_root}/objects.json")

    if args.model_name == "ACGan":
        netG = _netG(nz=args.z_dim)
        netD = _netD(num_classes=args.num_classes)
        ACGan_train(train_loader,test_condition_labels,netD,netG,device,args)
    elif args.model_name == "DCGan":
        netG = dc_netG(nz=args.z_dim)
        netD = dc_netD()
        DCGan_train(train_loader,test_condition_labels,netD,netG,device,args)
    else:
        raise ValueError("Invalid Model name (Excpet ACGan  or DCGan)")

if __name__ == '__main__':
    main()