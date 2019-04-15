from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader, Sampler
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision


class VggNetwork(nn.Module):
    def __init__(self, out):
        super(VggNetwork, self).__init__()
        self.vgg = torchvision.models.vgg11(pretrained=True)
        self.features = self.vgg.features
        self.avgpool = self.vgg.avgpool
        self.classifier = nn.Sequential(nn.Linear(25088, out))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]
