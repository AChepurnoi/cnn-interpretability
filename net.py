from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader, Sampler
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision


class VggNetwork(nn.Module):
    def __init__(self):
        super(VggNetwork, self).__init__()
        self.vgg = torchvision.models.vgg11(pretrained=True)
        self.adjust_conv = nn.Conv2d(1, 3, 1)
        self.features = self.vgg.features
        self.avgpool = self.vgg.avgpool
        self.classifier = nn.Sequential(nn.Linear(25088, 10))

    def forward(self, x):
        #         Conv to adjust 1 channel MNIST images to RGB required by vgg pretrained (will be removed later)
        x = self.adjust_conv(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]
