#%%
import argparse
import itertools
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from dataset import bair_robot_pushing_dataset
from models.lstm import gaussian_lstm, lstm
from models.vgg_64 import vgg_decoder, vgg_encoder
from utils import init_weights, kl_criterion, finn_eval_seq,imshowSeq,save_gif
from train_fixed_prior import parse_args,pred

def main():
    args = parse_args()

    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = f'cuda:{args.cuda_num}'
    else:
        device = 'cpu'
    
    print("device:",device)
    assert args.n_past + args.n_future <= 30 and args.n_eval <= 30
    assert 0 <= args.tfr and args.tfr <= 1
    assert 0 <= args.tfr_start_decay_epoch 
    assert 0 <= args.tfr_decay_step and args.tfr_decay_step <= 1

    if args.model_dir != '':
        # load model and continue training from checkpoint
        saved_model = torch.load('%s/model.pth' % args.model_dir)
        optimizer = args.optimizer
        model_dir = args.model_dir
        niter = args.niter
        cuda_num = args.cuda_num
        args = saved_model['args']
        args.optimizer = optimizer
        args.model_dir = model_dir
        args.cuda_num = cuda_num
        start_epoch = saved_model['last_epoch']

    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ------------ build the models  --------------

    if args.model_dir != '':
        decoder = saved_model['decoder']
        encoder = saved_model['encoder']
        frame_predictor = saved_model['frame_predictor']
        posterior = saved_model['posterior']
        frame_predictor.update_device(device)
        posterior.update_device(device)

    
    # --------- transfer to device ------------------------------------
    frame_predictor.to(device)
    posterior.to(device)
    encoder.to(device)
    decoder.to(device)

    # --------- load a dataset ------------------------------------
    validate_data = bair_robot_pushing_dataset(args, 'validate')

    validate_loader = DataLoader(validate_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)

    validate_iterator = iter(validate_loader)

    modules = {
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'encoder': encoder,
        'decoder': decoder,
    }
    # --------- training loop ------------------------------------

    frame_predictor.eval()
    encoder.eval()
    decoder.eval()
    posterior.eval()

    validate_seq, validate_cond = next(validate_iterator)

    #eval
    psnr_list = []
    for _ in range(len(validate_data) // args.batch_size):
        try:
            validate_seq, validate_cond = next(validate_iterator)
        except StopIteration:
            validate_iterator = iter(validate_loader)
            validate_seq, validate_cond = next(validate_iterator)
    
        pred_seq = pred(validate_seq, validate_cond, modules, args, device)
        _, _, psnr = finn_eval_seq(validate_seq[:,args.n_past:(args.n_past + args.n_future)], pred_seq)
        psnr_list.append(psnr)
        
    ave_psnr = np.mean(np.concatenate(psnr_list))

    # for i in range(3):
    #     save_gif(f"{args.model_dir}/predict{i}.gif",pred_seq[i].unsqueeze_(1),0.5)
    #     save_gif(f"{args.model_dir}/ground_true{i}.gif",validate_seq[:,args.n_past:(args.n_past + args.n_future)][i].unsqueeze_(1),0.5)
    print(('====================== validate psnr = {:.5f} ========================'.format(ave_psnr)))


if __name__ == '__main__':
    main()

