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
from utils import init_weights, kl_criterion, finn_eval_seq,imshowSeq

torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0, type=float, help='weight decay')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
    parser.add_argument('--batch_size', default=18, type=int, help='batch size')
    parser.add_argument('--log_dir', default='./logs/fp', help='base directory to save logs')
    parser.add_argument('--model_dir', default='', help='base directory to save logs')
    parser.add_argument('--data_root', default='./data', help='root directory for data')
    parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
    parser.add_argument('--use_scheduler', default=False, action='store_true', help='optimizer use_scheduler')
    parser.add_argument('--scheduler_factor', default=0.5, type=float, help='scheduler_factor used when optimizer is SGD')
    parser.add_argument('--min_lr', default=0, type=float, help='min learning rate used when optimizer is SGD')
    parser.add_argument('--scheduler_patience', default=10, type=int, help='scheduler_patience used when optimizer is SGD')
    parser.add_argument('--momentum', default=0.9,type=float, help='optimizer momentum')
    parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
    parser.add_argument('--tfr', type=float, default=1.0, help='teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_start_decay_epoch', type=int, default=0, help='The epoch that teacher forcing ratio become decreasing')
    parser.add_argument('--tfr_decay_step', type=float, default=0, help='The decay step size of teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_lower_bound', type=float, default=0, help='The lower bound of teacher forcing ratio for scheduling teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--kl_anneal_cyclical', default=False, action='store_true', help='use cyclical mode')
    parser.add_argument('--kl_anneal_ratio', type=float, default=0.5, help='The decay ratio of kl annealing')
    parser.add_argument('--kl_anneal_cycle', type=int, default=3, help='The number of cycle for kl annealing during training (if use cyclical mode)')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
    parser.add_argument('--n_eval', type=int, default=15, help='number of frames to predict at eval time')
    parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
    parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--z_dim', type=int, default=64, help='dimensionality of z_t')
    parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading threads')
    parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
    parser.add_argument('--cuda', default=True, action='store_true')
    parser.add_argument('--cuda_num', type=int, default=0)  

    args = parser.parse_args()
    return args

mse_criterion = nn.MSELoss()

def train(x, cond, modules, optimizer, kl_anneal, args,device):
    modules['frame_predictor'].zero_grad()
    modules['posterior'].zero_grad()
    modules['encoder'].zero_grad()
    modules['decoder'].zero_grad()

    # initialize the hidden state.
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()
    mse = 0
    kld = 0
    use_teacher_forcing = True if random.random() < args.tfr else False
    seq = [modules['encoder'](x[:,i].to(device)) for i in range(args.n_past + args.n_future)]
    for i in range(1, args.n_past + args.n_future):
        target = seq[i][0]
        if args.last_frame_skip or i < args.n_past:
            h, skip = seq[i - 1]
        else:
            if random.random() < args.tfr:
                h = seq[i- 1][0]
            else :
                h = modules['encoder'](x_pred.to(device))[0]
        z_t,mu,logvar = modules['posterior'](target)
        h_pred = modules['frame_predictor'](torch.cat([h,z_t,cond[:,i].to(device)],1))
        x_pred = modules['decoder']([h_pred,skip])
        mse += mse_criterion(x_pred,x[:,i].to(device))
        kld += kl_criterion(mu,logvar,args)

    beta = kl_anneal.get_beta()
    loss = mse + kld * beta
    loss.backward()

    optimizer.step()

    return loss.detach().cpu().numpy() / (args.n_past + args.n_future), mse.detach().cpu().numpy() / (args.n_past + args.n_future), kld.detach().cpu().numpy() / (args.n_future + args.n_past)

def pred(x, cond, modules, args ,device):
    pred_seq = None
    with torch.no_grad():
        for i in range(1, args.n_past + args.n_future):
            target = modules['encoder'](x[:,i].to(device))[0] if i < args.n_past else modules['encoder'](x_pred.to(device))[0]
            if args.last_frame_skip or i < args.n_past:
                h, skip = modules['encoder'](x[:,i - 1].to(device))
            else:
                h = h_t
            z_t,mu,logvar = modules['posterior'](target)
            h_pred = modules['frame_predictor'](torch.cat([h,z_t,cond[:,i].to(device)],1))
            x_pred = modules['decoder']([h_pred,skip])
            h_t = target

            if i >= args.n_past:
                pred_seq = torch.cat((pred_seq,x_pred.unsqueeze(0)),0) if pred_seq is not None else x_pred.unsqueeze(0)
    return pred_seq.transpose_(0,1)

def plot_pred(validate_seq, validate_cond, modules,epochs,args,device):
    pred_seq = pred(validate_seq, validate_cond, modules, args, device)
    imshowSeq(validate_seq[0,args.n_past:(args.n_past + args.n_future)],pred_seq[0],f"epochs: {str(epochs)}",f"{args.log_dir}/img/{epochs}.png")

class kl_annealing():
    def __init__(self, args):
        super().__init__()
        self.cyclical = args.kl_anneal_cyclical
        self.ratio = args.kl_anneal_ratio
        self.cycle = args.kl_anneal_cycle if self.cyclical else 1
        self.period = args.niter / self.cycle
        self.step = 1 / (self.period * self.ratio)
        self.v = 0
        self.i = 0

    def update(self):
        self.v += self.step
        self.i += 1
        if self.v >= 1:
            if self.i >= self.period:
                self.v = 0
                self.i = 0
            else :
                self.v = 1
    
    def get_beta(self):
        return self.v


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
        saved_model = torch.load('%s/model_each_epoch.ckpt' % args.model_dir)
        optimizer = args.optimizer
        model_dir = args.model_dir
        niter = args.niter
        cuda_num = args.cuda_num
        args = saved_model['args']
        args.optimizer = optimizer
        args.model_dir = model_dir
        args.cuda_num = cuda_num
        args.log_dir = '%s/continued' % args.log_dir
        start_epoch = saved_model['last_epoch']
    else:
        name = 'rnn_size=%d-predictor-posterior-rnn_layers=%d-%d-n_past=%d-n_future=%d-lr=%.4f-g_dim=%d-z_dim=%d-last_frame_skip=%s-beta=%.7f'\
            % (args.rnn_size, args.predictor_rnn_layers, args.posterior_rnn_layers, args.n_past, args.n_future, args.lr, args.g_dim, args.z_dim, args.last_frame_skip, args.beta)

        args.log_dir = '%s/%s' % (args.log_dir, name)
        niter = args.niter
        start_epoch = 0

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs('%s/gen/' % args.log_dir, exist_ok=True)
    os.makedirs('%s/img/' % args.log_dir, exist_ok=True)

    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)    
    torch.cuda.manual_seed_all(args.seed)

    if os.path.exists('./{}/train_record.log'.format(args.log_dir)):
        os.remove('./{}/train_record.log'.format(args.log_dir))
    
    print(args)

    logging.basicConfig(filename='./{}/train_record.log'.format(args.log_dir), encoding='utf-8', level=logging.INFO)
    logging.info('args: {}\n'.format(args))

    # ------------ build the models  --------------

    if args.model_dir != '':
        frame_predictor = saved_model['frame_predictor']
        posterior = saved_model['posterior']
        frame_predictor.update_device(device)
        posterior.update_device(device)
    else:
        frame_predictor = lstm(args.g_dim+args.z_dim+7, args.g_dim, args.rnn_size, args.predictor_rnn_layers, args.batch_size, device)
        posterior = gaussian_lstm(args.g_dim, args.z_dim, args.rnn_size, args.posterior_rnn_layers, args.batch_size, device)
        frame_predictor.apply(init_weights)
        posterior.apply(init_weights)
            
    if args.model_dir != '':
        decoder = saved_model['decoder']
        encoder = saved_model['encoder']
    else:
        encoder = vgg_encoder(args.g_dim)
        decoder = vgg_decoder(args.g_dim)
        encoder.apply(init_weights)
        decoder.apply(init_weights)
    
    # --------- transfer to device ------------------------------------
    frame_predictor.to(device)
    posterior.to(device)
    encoder.to(device)
    decoder.to(device)

    # --------- load a dataset ------------------------------------
    train_data = bair_robot_pushing_dataset(args, 'train')
    validate_data = bair_robot_pushing_dataset(args, 'validate')
    train_loader = DataLoader(train_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)
    train_iterator = iter(train_loader)

    validate_loader = DataLoader(validate_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)

    validate_iterator = iter(validate_loader)

    # ---------------- optimizers ----------------

    params = list(frame_predictor.parameters()) + list(posterior.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
    scheduler = None
    if args.optimizer == 'adam':
        optimizer = optim.Adam(params, lr=args.lr, betas=(args.beta1, 0.999),weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(params, lr=args.lr, momentum=args.momentum,weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum,weight_decay=args.weight_decay)
    else:
        raise ValueError('Unknown optimizer: %s' % args.optimizer)
    
    if args.model_dir != '' and 'optimizer' in saved_model:
        optimizer.load_state_dict(saved_model['optimizer'])
    
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.scheduler_factor, patience=args.scheduler_patience, min_lr=args.min_lr,verbose=True)

    kl_anneal = kl_annealing(args)

    modules = {
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'encoder': encoder,
        'decoder': decoder,
    }
    # --------- training loop ------------------------------------

    progress = tqdm(total=args.niter)
    best_val_psnr = 0
    for epoch in range(start_epoch, start_epoch + niter):
        frame_predictor.train()
        posterior.train()
        encoder.train()
        decoder.train()

        epoch_loss = 0
        epoch_mse = 0
        epoch_kld = 0

        for i in range(args.epoch_size):
            try:
                seq, cond = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                seq, cond = next(train_iterator)
            
            loss, mse, kld = train(seq, cond, modules, optimizer, kl_anneal, args, device)
            epoch_loss += loss
            epoch_mse += mse
            epoch_kld += kld

            if i % 20 == 0:
                print(f"[epoch_size: {i}/{args.epoch_size}]","\tLoss",loss,"\tMSE",mse,"\tKLD",kld)
        
        if epoch >= args.tfr_start_decay_epoch:
            args.tfr -= args.tfr_decay_step
            if args.tfr < args.tfr_lower_bound:
                args.tfr = args.tfr_lower_bound

        
        kl_anneal.update()
        progress.update(1)

        logging.info(('[epoch: %02d] loss: %.5f | mse loss: %.5f | kld loss: %.5f | lr: %.5f'
            % (epoch, epoch_loss  / args.epoch_size, epoch_mse / args.epoch_size, epoch_kld / args.epoch_size, optimizer.param_groups[0]["lr"])))

        frame_predictor.eval()
        encoder.eval()
        decoder.eval()
        posterior.eval()

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

        if scheduler:
            scheduler.step(ave_psnr)

        logging.info(('====================== validate psnr = {:.5f} ========================'.format(ave_psnr)))

        torch.save({
                'encoder': encoder,
                'decoder': decoder,
                'frame_predictor': frame_predictor,
                'posterior': posterior,
                'optimizer': optimizer.state_dict(),
                'args': args,
                'last_epoch': epoch},
                '%s/model_each_epoch.ckpt' % (args.log_dir))

        if ave_psnr > best_val_psnr:
            best_val_psnr = ave_psnr
            # save the model
            torch.save({
                'encoder': encoder,
                'decoder': decoder,
                'frame_predictor': frame_predictor,
                'posterior': posterior,
                'optimizer': optimizer.state_dict(),
                'args': args,
                'last_epoch': epoch},
                '%s/model.pth' % args.log_dir)
            
        if epoch % 10 == 0:
            torch.save({
                'encoder': encoder,
                'decoder': decoder,
                'frame_predictor': frame_predictor,
                'posterior': posterior,
                'optimizer': optimizer.state_dict(),
                'args': args,
                'last_epoch': epoch},
                '%s/model_each_ten_epoch.ckpt' % (args.log_dir))

        if epoch % 10 == 0:
            try:
                validate_seq, validate_cond = next(validate_iterator)
            except StopIteration:
                validate_iterator = iter(validate_loader)
                validate_seq, validate_cond = next(validate_iterator)

            plot_pred(validate_seq, validate_cond, modules, epoch, args ,device)

if __name__ == '__main__':
    main()
        
