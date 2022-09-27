# %%
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import RetinopathyLoader
from models import Retinopathy_Resnet
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
# %%


def predict_and_accuracy(model, test_loader, device):
    model.eval()
    accI_count = 0
    total_count = 0
    y_hat = None
    test_pbar = tqdm(test_loader, position=0, leave=True)

    for x, y in test_pbar:
        x, y = x.to(device), y.type(torch.LongTensor).to(device)
        with torch.no_grad():
            pred = model(x)
            pred = torch.argmax(pred, dim=1)
            y_hat = torch.cat((y_hat, pred)) if y_hat is not None else pred
            accI_count += (pred == y).sum()
            total_count += pred.shape[0]

    return y_hat.cpu(), (accI_count / total_count).cpu()
# %%


def show_confusion_matrix(y_hat, y, save_dir_path=None):
    mat = confusion_matrix(y_hat, y, normalize='all')
    sns.heatmap(mat, annot=True, cbar=False, fmt=".4g")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Normalized confusion matrix")
    if save_dir_path is not None:
        plt.savefig(f"{save_dir_path}/confusion_matrix.jpg")
    plt.show()


# %%
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32
    load_model_path = "../../final/resnet18_pretrained/best_model.ckpt"
    image_dir_path = "../../data"

    test_transforms_func = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = RetinopathyLoader(
        image_dir_path, "test", test_transforms_func)

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    model = Retinopathy_Resnet(18, False).to(device)
    model.load_state_dict(torch.load(load_model_path, map_location=device))

    print(f"load model path : {load_model_path}")
    y_hat, accu = predict_and_accuracy(model, test_loader, device)

    print(f"Test Accuracy: {np.array(accu):.2%}", )
    show_confusion_matrix(y_hat, test_dataset.label)

# %%
