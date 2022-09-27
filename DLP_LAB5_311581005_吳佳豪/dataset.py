#%%
import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

default_transforms = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
#%%
class Iclver_dataset(Dataset):
    def __init__(self, img_dir, json_path, object_path, transform=default_transforms):
        self.img_dir = img_dir
        self.json_path = json_path
        self.object_path = object_path
        self.transform = transform
        self.img_names = []
        self.img_onehot_labels = []
        with open(object_path, 'r') as f:
            self.classes = json.load(f)
        with open(json_path, 'r') as f:
            train_datas = json.load(f)
            for img_name,img_label_texts in train_datas.items():
                self.img_names.append(img_name)
                onehot = torch.zeros(len(self.classes))
                for text in img_label_texts:
                    onehot[self.classes[text]] = 1
                self.img_onehot_labels.append(onehot)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img=Image.open(os.path.join(self.img_dir,self.img_names[index])).convert('RGB')
        img=self.transform(img)
        label = self.img_onehot_labels[index]
        return img,label

#%%
def get_json_labels(json_path,object_json_path):
    print("test json path:",json_path)
    with open(object_json_path, 'r') as f:
        classes = json.load(f)
    with open(json_path, 'r') as f:
        labels_text = json.load(f)
    
    labels = torch.zeros(len(labels_text),len(classes))

    for index, texts in enumerate(labels_text):
        for text in texts:
            labels[index][classes[text]] = 1

    return labels

# %%
