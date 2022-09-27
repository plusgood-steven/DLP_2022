# %%
import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from dataloader import read_bci_data
from model import EEGNet, DeepConvNet

# %%


def same_seed(seed):
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def show_result(records, dir_path):
    plt.title("Train Loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    for record in records.items():
        epochs = np.array([e for e in range(len(record[1]["train_loss"]))])
        plt.plot(epochs, np.array(record[1]["train_loss"]), label=record[0])
    plt.legend()
    plt.savefig(f"{dir_path}/train_loss.jpg")
    plt.close()

    plt.title("Test Loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    for record in records.items():
        epochs = np.array([e for e in range(len(record[1]["test_loss"]))])
        plt.plot(epochs, np.array(record[1]["test_loss"]), label=record[0])
    plt.legend()
    plt.savefig(f"{dir_path}/test_loss.jpg")
    plt.close()

    plt.title("Train Accuracy Result")
    plt.xlabel("epochs")
    plt.ylabel("accuancy")
    for record in records.items():
        epochs = np.array([e for e in range(len(record[1]["train_accu"]))])
        plt.plot(epochs, np.array(record[1]["train_accu"]), label=record[0])
    plt.legend()
    plt.savefig(f"{dir_path}/train_accuancy.jpg")
    plt.close()

    plt.title("Test Accuracy Result")
    plt.xlabel("epochs")
    plt.ylabel("accuancy")
    for record in records.items():
        epochs = np.array([e for e in range(len(record[1]["test_accu"]))])
        plt.plot(epochs, np.array(record[1]["test_accu"]), label=record[0])
    plt.legend()
    plt.savefig(f"{dir_path}/test_accuancy.jpg")
    plt.close()

    plt.title("Loss comparison")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    for record in records.items():
        epochs = np.array([e for e in range(len(record[1]["train_loss"]))])
        plt.plot(epochs, np.array(
            record[1]["train_loss"]), label=record[0] + "_train")
        plt.plot(epochs, np.array(
            record[1]["test_loss"]), label=record[0] + "_test")
    plt.legend()
    plt.savefig(f"{dir_path}/loss_comparison.jpg")
    plt.close()

    plt.title("Accuracy comparison")
    plt.xlabel("epochs")
    plt.ylabel("accuancy")
    for record in records.items():
        epochs = np.array([e for e in range(len(record[1]["train_accu"]))])
        plt.plot(epochs, np.array(
            record[1]["train_accu"]), label=record[0] + "_train")
        plt.plot(epochs, np.array(
            record[1]["test_accu"]), label=record[0] + "_test")
    plt.legend()
    plt.savefig(f"{dir_path}/accuracy_comparison.jpg")
    plt.close()

# %%


class AddNoise:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, inputTensor):
        noise = torch.randn_like(inputTensor)
        return inputTensor + (self.sigma ** 0.5) * noise


transforms_func = transforms.Compose([
    AddNoise(0.5),
    transforms.RandomCrop((2, 750), padding=(20, 0, 20, 0),
                          pad_if_needed=False, fill=0, padding_mode='constant')
])

class BCIDataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''

    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

# %%


def trainer(train_loader, test_loader, model, config, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    
    writer = SummaryWriter()

    n_epochs, best_accu, best_loss, step, early_stop_count = config[
        'n_epochs'], 0, math.inf, 0, 0

    train_loss_records = []
    test_loss_records = []
    train_accu_records = []
    test_accu_records = []

    for epoch in range(n_epochs):
        model.train()
        loss_record = []

        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()

            x, y = transforms_func(x).to(device), y.type(
                torch.LongTensor).to(device)
            pred = model(x)
            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()
            step += 1
            loss_record.append(loss.detach().item())

            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record)/len(loss_record)
        train_loss_records.append(mean_train_loss)
        writer.add_scalar('Loss/train', mean_train_loss, step)

        # 計算每個epoch train loader accuracy
        train_accu, _ = predict_accuracy(
            model, train_loader, device, criterion)
        train_accu_records.append(train_accu)
        writer.add_scalar('Train Accuracy', train_accu, step)

        # 計算每個epoch test loader accuracy
        test_accu, mean_test_loss = predict_accuracy(
            model, test_loader, device, criterion)
        test_accu_records.append(test_accu)
        test_loss_records.append(mean_test_loss)
        writer.add_scalar('Test Accuracy', test_accu, step)

        print(
            f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f} Train Accuracy:{train_accu:.2%} Train loss: {mean_test_loss:.4f} Test Accuracy:{test_accu:.2%}')

        if best_accu < test_accu:
            best_accu = test_accu
            # Save your best accuracy model
            torch.save(model.state_dict(), config['save_path'])
            print(
                'Saving model with best accuracy {:.3f}...'.format(test_accu))
        if mean_train_loss < best_loss:
            best_loss = mean_train_loss
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return train_loss_records, train_accu_records, test_loss_records, test_accu_records
    return train_loss_records, train_accu_records, test_loss_records, test_accu_records


def predict_accuracy(model, test_loader, device, criterion=nn.CrossEntropyLoss):
    model.eval()
    accI_count = 0
    total_count = 0
    loss_record = []
    for x, y in test_loader:
        x, y = x.to(device), y.type(torch.LongTensor).to(device)
        with torch.no_grad():
            pred = model(x)
            loss = criterion(pred, y)
            pred = torch.argmax(pred, dim=1)
            accI_count += (pred == y).sum()
            total_count += pred.shape[0]
        loss_record.append(loss.item())

    return (accI_count / total_count).cpu(), sum(loss_record)/len(loss_record)


def train_all_model(config, train_loader, test_loader, device):
    # 訓練的模型與激勵函數
    models = {"EEGNet":EEGNet,"DeepConvNet":DeepConvNet}
    activation_functions = {"ReLU": nn.ReLU,
                            "LeakyReLU": nn.LeakyReLU, "ELU": nn.ELU}

    # save data path
    config["model_dir_path"] = config["dir_path"] + "/B{batch_size}_LR{learning_rate:.1e}".format(
        batch_size=config["batch_size"], learning_rate=config["learning_rate"])

    if not os.path.isdir(config["model_dir_path"]):
        os.makedirs(config["model_dir_path"])

    best_accu_file = open(
        config["model_dir_path"] + "/all_model_best_accu.txt", "w")
    best_accu_file.write("Batch_Size: {batch_size} LR: {learning_rate:.1e}\n".format(
        batch_size=config["batch_size"], learning_rate=config["learning_rate"]))
    info_record = {}
    for model_constructor in models.items():
        model_name = model_constructor[0]
        model_instance = model_constructor[1]

        for activation in activation_functions.items():
            function_name = activation[0]
            function_instance = activation[1]
            config["save_path"] = config["model_dir_path"] + \
                f"/{model_name}_{function_name}.ckpt"

            model = model_instance(function_instance).to(device)
            train_loss_record, train_accu_record, test_loss_record, test_accu_record = trainer(
                train_loader, test_loader, model, config, device)
            info_record[f'{model_name}_{function_name}'] = {"train_loss": train_loss_record,
                                                            "train_accu": train_accu_record, "test_loss": test_loss_record, "test_accu": test_accu_record}
            best_accu_file.write(
                f"{model_name}_{function_name}: Test Accuracy: {max(test_accu_record):.2%}\n")

    show_result(info_record, config["model_dir_path"])
    best_accu_file.close()


#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'n_epochs': 1000,     # Number of epochs.
    'batch_size': 128,
    'learning_rate': 1e-3,
    'early_stop': 400,  # If model has not improved for this many consecutive epochs, stop training.
    'dir_path': './result/test',  # model will be saved this dir.
    "activation_function": nn.ReLU
}
same_seed(123456)
print("device :", device)
# %%
x_train, y_train, x_test, y_test = read_bci_data()

train_dataset, test_dataset = BCIDataset(
    x_train, y_train), BCIDataset(x_test, y_test)

# Pytorch data loader loads pytorch dataset into batches.
train_loader = DataLoader(
    train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(
    test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

print("batch_size", config['batch_size'])
train_all_model(config, train_loader, test_loader, device)

# %%