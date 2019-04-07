from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader, Sampler
import torch.nn.functional as F
import numpy as np
import torch
import tqdm
from torchvision import datasets, transforms
import torchvision


class Trainer:
    def __init__(self, net, config):
        net.train()
        net.to(config['DEVICE'])
        self.device = config['DEVICE']
        self.net = net
        self.config = config
        self.optimizer = torch.optim.Adam(net.trainable_params(), lr=0.001)

    def run(self, dataloader, epoch=1):
        print(">> Running trainer")
        for epoch in range(epoch):
            print(">>> Epoch %s" % epoch)
            for idx, (image, target) in enumerate(tqdm.tqdm_notebook(dataloader, ascii=True)):
                image, target = image.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                predict = self.net(image)
                loss = F.nll_loss(predict, target)
                loss.backward()
                self.optimizer.step()
                if self.config['DEBUG'] == True:
                    break
            print("Trainer epoch finished")


class Evaluation:
    def __init__(self, net, config):
        net.eval()
        net.to(config['DEVICE'])
        self.device = config['DEVICE']
        self.net = net
        self.config = config

    def run(self, dataloader):
        print(">> Running Evaluation")
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for idx, (image, target) in enumerate(tqdm.tqdm_notebook(dataloader, ascii=True)):
                image, target = image.to(self.device), target.to(self.device)
                predict = self.net(image)
                test_loss += F.nll_loss(predict, target, reduction='sum').item()  # sum up batch loss
                pred = predict.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                if self.config['DEBUG'] == True:
                    break

        test_loss /= len(dataloader.dataset)
        print("Evaluation finished")
        return {
            'loss': test_loss,
            'accuracy': 100. * correct / len(dataloader.dataset)
        }
