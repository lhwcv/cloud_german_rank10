import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.senet import se_resnet50
from backbone.inceptionresnetv2 import  inceptionresnetv2
from backbone.xception import  xception
class GermanNetSE50(nn.Module):
    def __init__(self, only_fea=False):
        super(GermanNetSE50, self).__init__()
        base = se_resnet50()
        self.layer1 = nn.Sequential(
            nn.Conv2d(18, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True),
        )
        self.layer2 = base.layer1
        self.layer3 = base.layer2
        self.layer4 = base.layer3
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, 17)
        self.only_fea = only_fea

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        if self.only_fea:
            return x
        return self.fc(x)


class GermanNetXcep(nn.Module):
    def __init__(self, return_feat=False):
        super(GermanNetXcep,self).__init__()
        model =xception(pretrained='imagenet')
        model.conv1 = nn.Conv2d(18, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.last_linear = nn.Linear(2048, 17)
        self.model = model
    def forward(self, x):
        return self.model(x)

class GermanNetIncepRes(nn.Module):
    def __init__(self, return_feat=False):
        super(GermanNetIncepRes,self).__init__()
        self.model =inceptionresnetv2(pretrained='imagenet')
        self.model.conv2d_1a.conv = nn.Conv2d(18, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2080, 17)
    def forward(self, input):
        x = self.model.conv2d_1a(input)
        x = self.model.conv2d_2a(x)
        x = self.model.conv2d_2b(x)
        x = self.model.maxpool_3a(x)
        x = self.model.conv2d_3b(x)
        x = self.model.conv2d_4a(x)
        x = self.model.maxpool_5a(x)
        x = self.model.mixed_5b(x)
        x = self.model.repeat(x)
        x = self.model.mixed_6a(x)
        x = self.model.repeat_1(x)
        x = self.model.mixed_7a(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x