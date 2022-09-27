import torch
import numpy as np
import matplotlib.pyplot as plt

def same_seed(seed):
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def show_result(records,save_dir_path):
    plt.title("Train Loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    for record in records.items():
        epochs = np.array([e for e in range(len(record[1]["train_loss"]))])
        plt.plot(epochs, np.array(record[1]["train_loss"]), label=record[0])
    plt.legend()
    plt.savefig(f"{save_dir_path}/train_loss.jpg")
    plt.close()

    plt.title("Test Loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    for record in records.items():
        epochs = np.array([e for e in range(len(record[1]["test_loss"]))])
        plt.plot(epochs, np.array(record[1]["test_loss"]), label=record[0])
    plt.legend()
    plt.savefig(f"{save_dir_path}/test_loss.jpg")
    plt.close()

    plt.title("Train Accuracy Result")
    plt.xlabel("epochs")
    plt.ylabel("accuancy")
    for record in records.items():
        epochs = np.array([e for e in range(len(record[1]["train_accu"]))])
        plt.plot(epochs, np.array(record[1]["train_accu"]), label=record[0])
    plt.legend()
    plt.savefig(f"{save_dir_path}/train_accuacy.jpg")
    plt.close()

    plt.title("Test Accuracy Result")
    plt.xlabel("epochs")
    plt.ylabel("accuancy")
    for record in records.items():
        epochs = np.array([e for e in range(len(record[1]["test_accu"]))])
        plt.plot(epochs, np.array(record[1]["test_accu"]), label=record[0])
    plt.legend()
    plt.savefig(f"{save_dir_path}/test_accuacy.jpg")
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
    plt.savefig(f"{save_dir_path}/loss_comparison.jpg")
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
    plt.savefig(f"{save_dir_path}/accuracy_comparison.jpg")
    plt.close()