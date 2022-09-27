import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dataloader import RetinopathyLoader
from process import same_seed, show_result
from models import Retinopathy_Resnet
import logging
import argparse


def train(train_loader, test_loader, model, config, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9, weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4, verbose=True,min_lr=1e-3)

    #writer = SummaryWriter()
    logging.basicConfig(
        filename=config['dir_path'] + '/train.log', format='%(levelname)s:%(message)s', encoding='utf-8', level=logging.INFO)
    logging.info(
        f"Model:{model.model_name} Optimizer:{optimizer.__class__.__name__} epochs: {config['n_epochs']} learning_rate: {config['learning_rate']} batch_size: {config['batch_size']} weight_decay: {config['weight_decay']}")
    logging.info("training")

    n_epochs, best_accu, best_loss, step, early_stop_count = config[
        'n_epochs'], 0, math.inf, 0, 0

    train_loss_records = []
    test_loss_records = []
    train_accu_records = []
    test_accu_records = []

    for epoch in range(n_epochs):
        model.train()
        loss_record = []
        train_accI_count = 0
        train_total_count = 0

        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()

            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            train_pred = torch.argmax(pred, dim=1)
            train_accI_count += (train_pred == y).sum()
            train_total_count += pred.shape[0]

            loss.backward()
            optimizer.step()
            step += 1
            loss_record.append(loss.detach().item())

            train_pbar.set_description(
                f'Epoch [{epoch+1}/{n_epochs}] training')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record)/len(loss_record)
        train_loss_records.append(mean_train_loss)
        #writer.add_scalar('Loss/train', mean_train_loss, step)

        train_accu = (train_accI_count / train_total_count).cpu()
        train_accu_records.append(train_accu)
        #writer.add_scalar('Train Accuracy', train_accu, step)

        # 計算每個epoch test loader accuracy
        test_accu, mean_test_loss = predict_accuracy(
            model, test_loader, device, criterion)
        test_accu_records.append(test_accu)
        test_loss_records.append(mean_test_loss)
        #writer.add_scalar('Test Accuracy', test_accu, step)

        scheduler.step(mean_test_loss)

        print(
            f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f} Train Accuracy:{train_accu:.2%} Test loss: {mean_test_loss:.4f} Test Accuracy:{test_accu:.2%}')
        logging.info(
            f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f} Train Accuracy:{train_accu:.2%} Test loss: {mean_test_loss:.4f} Test Accuracy:{test_accu:.2%}')

        if best_accu < test_accu:
            best_accu = test_accu
            # Save your best accuracy model
            torch.save(model.state_dict(),
                       f"{config['dir_path']}/best_model.ckpt")
            print(
                'Saving model with best accuracy {:.3f}...'.format(test_accu))
            logging.info(
                'Saving model with best accuracy {:.3f}...'.format(test_accu))

        if epoch % 3 == 0:
            torch.save(model.state_dict(),
                       f"{config['dir_path']}/checkpoint{epoch}.ckpt")

        if mean_test_loss < best_loss:
            best_loss = mean_test_loss
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return train_loss_records, train_accu_records, test_loss_records, test_accu_records

    return train_loss_records, train_accu_records, test_loss_records, test_accu_records


def predict_accuracy(model, test_loader, device, criterion=nn.CrossEntropyLoss):
    model.eval()
    logging.info("evaling")
    accI_count = 0
    total_count = 0
    loss_record = []
    test_pbar = tqdm(test_loader, position=0, leave=True)

    for x, y in test_pbar:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x)
            loss = criterion(pred, y)
            pred = torch.argmax(pred, dim=1)
            accI_count += (pred == y).sum()
            total_count += pred.shape[0]
        loss_record.append(loss.item())
        test_pbar.set_description(f'testing')
        test_pbar.set_postfix({'loss': loss.detach().item()})

    return (accI_count / total_count).cpu(), sum(loss_record)/len(loss_record)


train_transforms_func = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=30, scale=(0.9, 1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms_func = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer_size",default=18,type=int)
    parser.add_argument("--epochs",default=50,type=int)
    parser.add_argument("--batch_size",default=64,type=int)
    parser.add_argument("--lr",default=1e-2,type=float)
    parser.add_argument("--weight_decay",default=1e-3,type=float)
    parser.add_argument("--early_stop",default=20,type=int)
    parser.add_argument("--dir_path",default="./results",type=str)
    parser.add_argument("--gpu",default="cuda",type=str)
    parser.add_argument("--image_dir_path",default="../data",type=str)
    parser.add_argument("--pretrained",default=True,action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    config = {
        'n_epochs': args.epochs,     # Number of epochs.
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'early_stop': args.early_stop,
        'dir_path': args.dir_path
    }
    device = args.gpu if torch.cuda.is_available() else 'cpu'
    same_seed(123456)
    print("device :", device)
    # %%
    train_dataset, test_dataset = RetinopathyLoader(
        args.image_dir_path, "train", transform=train_transforms_func), RetinopathyLoader(args.image_dir_path, "test", test_transforms_func)

    # Pytorch data loader loads pytorch dataset into batches.
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True, num_workers=4)

    # %%
    info_records = {}

    config['dir_path'] = f"./{args.dir_path}/resnet{args.layer_size}_pretrained" if args.pretrained else f"./{args.dir_path}/resnet{args.layer_size}"

    if not os.path.isdir(config["dir_path"]):
        os.makedirs(config["dir_path"])

    Resnet_model = Retinopathy_Resnet(args.layer_size, pretrained=args.pretrained).to(device)

    train_loss_records, train_accu_records, test_loss_records, test_accu_records = train(
        train_loader, test_loader, Resnet_model, config, device)
    info_records[Resnet_model.model_name] = {"train_loss": train_loss_records,
                                                    "test_loss": test_loss_records, "train_accu": train_accu_records, "test_accu": test_accu_records}
    show_result(info_records, config["dir_path"])

# %%