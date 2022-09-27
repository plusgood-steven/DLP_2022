#%%
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import weights_init
from evaluator import evaluation_model
from tqdm import tqdm
from model import _netD, _netG, dc_netD,dc_netG
import logging
import copy
from torchvision.utils import save_image

def ACGan_train(train_loader, test_conditions, netD: _netD, netG: _netG, device, args):
    netG.to(device)
    netD.to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    num_classes = args.num_classes
    batch_size = args.batch_size
    z_dim = args.z_dim

    evaluation = evaluation_model(device=device)
    # setup optimizer
    optimizerD = optim.Adam(
        netD.parameters(), lr=args.lr, betas=(args.beta, 0.999))
    optimizerG = optim.Adam(
        netG.parameters(), lr=args.lr, betas=(args.beta, 0.999))
    # loss functions
    dis_criterion = nn.BCELoss().to(device)
    aux_criterion = nn.CrossEntropyLoss().to(device)


    best_accuracy = 0
    for epoch in range(args.epochs):
        progress = tqdm(train_loader)
        loss_d = 0
        loss_g = 0
        realfake_loss_ratio = args.realfake_loss_ratio
        classes_loss_ratio = args.classes_loss_ratio
        realfake_loss_decay = args.realfake_loss_decay * (epoch - args.realfake_start_epoch) if epoch > args.realfake_start_epoch else 0
        for _, (imgs, labels) in enumerate(progress):
            netG.train()
            netD.train()

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()

            aux_label = copy.deepcopy(labels).to(device) if  args.label_smoothing == 0 else label_smoothing_func(labels, args.label_smoothing).to(device)

            predict_realfake, predict_classes = netD(imgs.to(device))  # return realfake, classes
            dis_label = torch.ones(batch_size).to(device)

            dis_errD_real = dis_criterion(predict_realfake, dis_label)
            aux_errD_real = aux_criterion(predict_classes, aux_label)
            errD_real = dis_errD_real * (realfake_loss_ratio - realfake_loss_decay) + aux_errD_real * (classes_loss_ratio + realfake_loss_decay)

            # train with fake
            noise = torch.normal(0, 1, (batch_size, z_dim - num_classes, 1, 1))
            c_labels = copy.deepcopy(labels).resize_((batch_size, num_classes, 1, 1))
            z = Variable(torch.cat((c_labels,noise), dim=1))
            dis_label = torch.zeros(batch_size).to(device)

            fake = netG(z.to(device))
            predict_realfake, predict_classes = netD(fake.detach())
            dis_errD_fake = dis_criterion(predict_realfake, dis_label)
            aux_errD_fake = aux_criterion(predict_classes, aux_label)
            errD_fake = dis_errD_fake * (realfake_loss_ratio - realfake_loss_decay) + aux_errD_fake * (classes_loss_ratio + realfake_loss_decay)

            errD = (errD_real + errD_fake) * 0.5

            errD.backward()
            optimizerD.step()
            
            loss_d += errD.item()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            for _ in range(args.generator_iter_loop):
                netG.zero_grad()

                noise = torch.normal(0, 1, (batch_size, z_dim - num_classes, 1, 1))
                c_labels = copy.deepcopy(labels).resize_((batch_size, num_classes, 1, 1))
                z = Variable(torch.cat((c_labels,noise), dim=1))

                fake = netG(z.to(device))
                # fake labels are real for generator cost
                dis_label = torch.ones(batch_size).to(device)

                predict_realfake, predict_classes = netD(fake)
                dis_errG = dis_criterion(predict_realfake, dis_label)
                aux_errG = aux_criterion(predict_classes, aux_label)
                errG = dis_errG * (realfake_loss_ratio - realfake_loss_decay) + aux_errG * (classes_loss_ratio + realfake_loss_decay)
                errG.backward()
                optimizerG.step()

            loss_g += errG.item()

            progress.set_description(
                f'Epoch [{epoch + 1}/{args.epochs}] training')

        # evaluate
        netG.eval()
        netD.eval()
        noise = torch.normal(0, 1, (len(test_conditions), z_dim - num_classes, 1, 1))
        c_labels = copy.deepcopy(test_conditions).resize_((len(test_conditions), num_classes, 1, 1))
        z = torch.cat((c_labels,noise), dim=1)
        with torch.no_grad():
            gen_imgs = netG(z.to(device))
        accuracy = evaluation.eval(gen_imgs, test_conditions)

        torch.save({
            'netG': netG,
            'netD': netD,
            'args': args,
            'last_epoch': epoch},
            '%s/model_each_epoch.ckpt' % (args.log_dir))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'netG': netG,
                'netD': netD,
                'args': args,
                'last_epoch': epoch},
                '%s/model.pth' % (args.log_dir))

        if epoch % 5 == 0:
            save_image(
                gen_imgs, fp=f"{args.log_dir}/img/epoch_{epoch}.png", nrow=8, normalize=True)

        print(f'Epoch[{epoch + 1}] loss_d: {loss_d / len(train_loader):.5f} loss_g: {loss_g / len(train_loader):.5f} accuracy: {accuracy:.2%} best accuracy: {best_accuracy:.2%}')
        logging.info(
            f'Epoch[{epoch + 1}] loss_d: {loss_d / len(train_loader):.5f} loss_g: {loss_g / len(train_loader):.5f} accuracy: {accuracy:.2%} best accuracy: {best_accuracy:.2%}')
        logging.info('----------------------------------------')


def DCGan_train(train_loader, test_conditions, netD: dc_netD, netG: dc_netG, device, args):
    netG.to(device)
    netD.to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    num_classes = args.num_classes
    batch_size = args.batch_size
    z_dim = args.z_dim

    evaluation = evaluation_model(device=device)
    # setup optimizer
    optimizerD = optim.Adam(
        netD.parameters(), lr=args.lr, betas=(args.beta, 0.999))
    optimizerG = optim.Adam(
        netG.parameters(), lr=args.lr, betas=(args.beta, 0.999))
    # loss functions
    dis_criterion = nn.BCELoss().to(device)

    best_accuracy = 0
    for epoch in range(args.epochs):
        progress = tqdm(train_loader)
        loss_d = 0
        loss_g = 0
        realfake_loss_ratio = args.realfake_loss_ratio
        realfake_loss_decay = args.realfake_loss_decay * (epoch - args.realfake_start_epoch) if epoch > args.realfake_start_epoch else 0
        for _, (imgs, labels) in enumerate(progress):
            netG.train()
            netD.train()

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()

            predict_realfake = netD(imgs.to(device),copy.deepcopy(labels).to(device))  # return realfake, classes
            dis_label = torch.ones(batch_size).to(device)

            dis_errD_real = dis_criterion(predict_realfake, dis_label)
            errD_real = dis_errD_real * (realfake_loss_ratio - realfake_loss_decay)

            # train with fake
            noise = torch.normal(0, 1, (batch_size, z_dim - num_classes, 1, 1))
            c_labels = copy.deepcopy(labels).resize_((batch_size, num_classes, 1, 1))
            z = Variable(torch.cat((c_labels,noise), dim=1))
            dis_label = torch.zeros(batch_size).to(device)

            fake = netG(z.to(device))
            predict_realfake= netD(fake.detach(),copy.deepcopy(labels).to(device))
            dis_errD_fake = dis_criterion(predict_realfake, dis_label)
            errD_fake = dis_errD_fake * (realfake_loss_ratio - realfake_loss_decay)

            errD = (errD_real + errD_fake) * 0.5

            errD.backward()
            optimizerD.step()
            
            loss_d += errD.item()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            for _ in range(args.generator_iter_loop):
                netG.zero_grad()

                noise = torch.normal(0, 1, (batch_size, z_dim - num_classes, 1, 1))
                c_labels = copy.deepcopy(labels).resize_((batch_size, num_classes, 1, 1))
                z = Variable(torch.cat((c_labels,noise), dim=1))

                fake = netG(z.to(device))
                # fake labels are real for generator cost
                dis_label = torch.ones(batch_size).to(device)

                predict_realfake = netD(fake,copy.deepcopy(labels).to(device))
                dis_errG = dis_criterion(predict_realfake, dis_label)
                errG = dis_errG * (realfake_loss_ratio - realfake_loss_decay)

                errG.backward()
                optimizerG.step()

            loss_g += errG.item()

            progress.set_description(
                f'Epoch [{epoch + 1}/{args.epochs}] training')

        # evaluate
        netG.eval()
        netD.eval()
        noise = torch.normal(0, 1, (len(test_conditions), z_dim - num_classes, 1, 1))
        c_labels = copy.deepcopy(test_conditions).resize_((len(test_conditions), num_classes, 1, 1))
        z = torch.cat((c_labels,noise), dim=1)
        with torch.no_grad():
            gen_imgs = netG(z.to(device))
        accuracy = evaluation.eval(gen_imgs, test_conditions)

        torch.save({
            'netG': netG,
            'netD': netD,
            'args': args,
            'last_epoch': epoch},
            '%s/model_each_epoch.ckpt' % (args.log_dir))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'netG': netG,
                'netD': netD,
                'args': args,
                'last_epoch': epoch},
                '%s/model.pth' % (args.log_dir))

        if epoch % 5 == 0:
            save_image(
                gen_imgs, fp=f"{args.log_dir}/img/epoch_{epoch}.png", nrow=8, normalize=True)

        print(f'Epoch[{epoch + 1}] loss_d: {loss_d / len(train_loader):.5f} loss_g: {loss_g / len(train_loader):.5f} accuracy: {accuracy:.2%} best accuracy: {best_accuracy:.2%}')
        logging.info(
            f'Epoch[{epoch + 1}] loss_d: {loss_d / len(train_loader):.5f} loss_g: {loss_g / len(train_loader):.5f} accuracy: {accuracy:.2%} best accuracy: {best_accuracy:.2%}')
        logging.info('----------------------------------------')

#%%
def label_smoothing_func(labels, smoothing=0.1):
    """
    Apply label smoothing to labels

    Args:
        labels: Tensor of shape [batch_size,N]
        smoothing: Label smoothing factor
    Returns:
        Tensor of shape [batch_size,N]
    """
    num_classes = len(labels[0])
    return (1 - smoothing) * labels + smoothing / num_classes

# %%