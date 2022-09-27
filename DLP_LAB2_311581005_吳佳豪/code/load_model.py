# %%
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataloader import read_bci_data
from model import EEGNet, DeepConvNet


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


def predict_accuracy(model, test_loader, device):
    model.eval()
    accI_count = 0
    total_count = 0
    for x, y in test_loader:
        x, y = x.to(device), y.type(torch.LongTensor).to(device)
        with torch.no_grad():
            pred = model(x)
            pred = torch.argmax(pred, dim=1)
            accI_count += (pred == y).sum()
            total_count += pred.shape[0]

    return (accI_count / total_count).cpu()


#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = {
    'n_epochs': 1000,     # Number of epochs.
    'batch_size': 1080,
    'learning_rate': 1e-3,
    # If model has not improved for this many consecutive epochs, stop training.
    'early_stop': 400,
    "activation_function": nn.ELU
}

x_train, y_train, x_test, y_test = read_bci_data()

train_dataset, test_dataset = BCIDataset(
    x_train, y_train), BCIDataset(x_test, y_test)

# Pytorch data loader loads pytorch dataset into batches.
train_loader = DataLoader(
    train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(
    test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

load_model_path = "./result/transforms_addAll/B128_LR1.0e-03/EEGNet_ReLU.ckpt"
model = EEGNet(nn.ReLU).to(device)
model.load_state_dict(torch.load(load_model_path))
print(f"load model path : {load_model_path}")
print(f"Test Accuracy: {np.array(predict_accuracy(model, test_loader, device)):.2%}", )

# %%
