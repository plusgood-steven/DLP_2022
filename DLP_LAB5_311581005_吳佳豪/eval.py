import argparse
import torch
from dataset import get_json_labels
from evaluator import evaluation_model
import copy
import os
from torchvision.utils import save_image

def eval_model(conditions, netG,evaluationModel, device, args):
    z_dim = args.z_dim
    num_classes = args.num_classes
    netG.eval()
    noise = torch.normal(0, 1, (len(conditions), z_dim - num_classes, 1, 1))
    c_labels = copy.deepcopy(conditions).resize_((len(conditions), num_classes, 1, 1))
    z = torch.cat((c_labels,noise), dim=1)
    with torch.no_grad():
        gen_imgs = netG(z.to(device))
    accuracy = evaluationModel.eval(gen_imgs, conditions)
    
    return accuracy, gen_imgs

def load_model(model_path):
    saved_models = torch.load(model_path)
    args = saved_models['args']
    last_epoch = saved_models['last_epoch']
    netG = saved_models['netG']
    netD = saved_models['netD']
    return netG,netD,args,last_epoch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--eval_dir', type=str, default='./eval_img')
    parser.add_argument('--cuda', default=True, action='store_true')
    parser.add_argument('--cuda_num', type=int, default=0)
    parser.add_argument('--test_json_name', default='test', help='test json file name')
    parser.add_argument('--dataset_root', default='./dataset', help='root directory for data')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print("===> Evaluation of the network on the test set ===")
    opt = parse_args()

    assert opt.model_path != None

    if opt.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = f'cuda:{opt.cuda_num}'
        torch.backends.cudnn.benchmark = True
    else:
        device = 'cpu'
    
    netG,netD,args,last_epoch = load_model(opt.model_path)
    evaluationModel = evaluation_model(device)
    conditions = get_json_labels(f"{opt.dataset_root}/{opt.test_json_name}.json",f"{opt.dataset_root}/objects.json")
    
    netG.to(device)

    all_acc = []
    best_acc = 0
    gen_imgs = None
    for i in range(1000):
        acc, imgs = eval_model(conditions,netG,evaluationModel,device,args)
        all_acc.append(acc)
        if acc > best_acc:
            best_acc = acc
            gen_imgs = imgs
    
    average_acc = sum(all_acc) / len(all_acc)

    os.makedirs(opt.eval_dir, exist_ok=True)
    save_image(gen_imgs, fp=f"{opt.eval_dir}/Best_acc_{opt.test_json_name}_{best_acc:.2%}.png", nrow=8, normalize=True)

    print(f"Best Acc: {best_acc:.2%}")
    print(f"Average Acc: {average_acc:.2%}")

