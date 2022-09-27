#%%
from torchvision import models
import torch.nn as nn

class Retinopathy_Resnet(nn.Module):
    def __init__(self,layer_size,pretrained=False):
        super(Retinopathy_Resnet, self).__init__()
        self.layer_size = layer_size if layer_size == 18 else 50
        self.pretrained = pretrained
        self.model = models.resnet18(pretrained=self.pretrained) if self.layer_size == 18 else models.resnet50(pretrained=self.pretrained)
        self.model_name = f"resnet{self.layer_size}_pretrained" if self.pretrained else f"resnet{self.layer_size}"
        if self.layer_size == 18 :
            self.model.fc = nn.Linear(512,5)
        else:
            self.model.fc = nn.Linear(2048,5)
        
    def forward(self,x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    test = Retinopathy_Resnet(18,True)
#%%