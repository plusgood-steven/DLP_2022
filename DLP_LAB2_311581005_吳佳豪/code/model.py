# %%
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, activationFunction):
        super(EEGNet, self).__init__()

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1,affine=True, track_running_stats=True)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1),stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1,affine=True, track_running_stats=True),
            activationFunction(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.25)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1,affine=True, track_running_stats=True),
            activationFunction(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.25)
        )
        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=736, out_features=2, bias=True)
        )

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = self.classify(x)
        return x


class DeepConvNet(nn.Module):
    def __init__(self, activationFunction):
        super(DeepConvNet, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5), bias=True),
            nn.Conv2d(25, 25, kernel_size=(2, 1), bias=True),
            nn.BatchNorm2d(25, eps=1e-5, momentum=0.1),
            activationFunction(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5),

            nn.Conv2d(25, 50, kernel_size=(1, 5), bias=True),
            nn.BatchNorm2d(50, eps=1e-5, momentum=0.1),
            activationFunction(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5),

            nn.Conv2d(50, 100, kernel_size=(1, 5), bias=True),
            nn.BatchNorm2d(100, eps=1e-5, momentum=0.1),
            activationFunction(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5),

            nn.Conv2d(100, 200, kernel_size=(1, 5), bias=True),
            nn.BatchNorm2d(200, eps=1e-5, momentum=0.1),
            activationFunction(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5),
        )
        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=8600, out_features=2),
        )

    def forward(self, x):
        x = self.model(x)
        x = self.classify(x)
        return x
# %%
