import torch
import torch.nn as nn

class _netG(nn.Module):
    def __init__(self, nz):
        super(_netG, self).__init__()
        self.nz = nz

        # first linear layer
        self.fc1 = nn.Linear(self.nz, 384)
        # Transposed Convolution 2
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(384, 192, 4, 1, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        # Transposed Convolution 3
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(192, 96, 4, 2, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(96, 48, 4, 2, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(True),
        )
        # Transposed Convolution 5
        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, 2, 1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(True),
        )

        # Transposed Convolution 6
        self.tconv6 = nn.Sequential(
            nn.ConvTranspose2d(24, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        input = input.view(-1, self.nz)
        fc1 = self.fc1(input)
        fc1 = fc1.view(-1, 384, 1, 1)
        tconv2 = self.tconv2(fc1)
        tconv3 = self.tconv3(tconv2)
        tconv4 = self.tconv4(tconv3)
        tconv5 = self.tconv5(tconv4)
        tconv6 = self.tconv6(tconv5)
        output = tconv6
        return output


class _netD(nn.Module):
    def __init__(self, num_classes=24):
        super(_netD, self).__init__()
        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 6
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        self.fc_dis = nn.Linear(5*5*512, 1)
        self.fc_aux = nn.Linear(5*5*512, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        flat6 = conv6.view(-1, 5*5*512) #5*5*512
        fc_dis = self.fc_dis(flat6)
        fc_aux = self.fc_aux(flat6)
        classes = self.sigmoid(fc_aux)
        realfake = self.sigmoid(fc_dis).view(-1, 1).squeeze(1)
        return realfake, classes



class dc_netG(nn.Module):
    def __init__(self, nz):
        super(dc_netG, self).__init__()
        self.nz = nz

        # Transposed Convolution 1
        self.tconv1 = nn.Sequential(
            nn.ConvTranspose2d(self.nz, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )
        # Transposed Convolution 2
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        # Transposed Convolution 3
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        # Transposed Convolution 5
        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        input = input.view(-1, self.nz, 1, 1)
        tconv1 = self.tconv1(input)
        tconv2 = self.tconv2(tconv1)
        tconv3 = self.tconv3(tconv2)
        tconv4 = self.tconv4(tconv3)
        tconv5 = self.tconv5(tconv4)
        output = tconv5
        return output

class dc_netD(nn.Module):
    def __init__(self):
        super(dc_netD, self).__init__()

        self.addConditionInfo=nn.Sequential(
            nn.Linear(24, 64 * 64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3 + 1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Convolution 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Convolution 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Convolution 5
        self.conv5 = nn.Conv2d(512, 1, 4, 1, 0, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input,c):
        c = self.addConditionInfo(c).view(-1,1,64,64)
        input = torch.cat((input,c),dim=1)
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        realfake = self.sigmoid(conv5).view(-1, 1).squeeze(1)
        return realfake

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

