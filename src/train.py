from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader, Sampler
import torch.nn.functional as F
import numpy as np
import torch
import tqdm
from torchvision import datasets, transforms
import torchvision

from src.utils import save_checkpoint


class Trainer:
    def __init__(self, net, config):
        net.train()
        net.to(config['DEVICE'])
        self.device = config['DEVICE']
        self.net = net
        self.config = config
        self.optimizer = torch.optim.Adam(net.trainable_params(), lr=0.001)
        self.loss = nn.CrossEntropyLoss()
        self.losses = []

    def run(self, dataloader, epochs=1):
        print(">> Running trainer")
        for epoch in range(epochs):
            print(">>> Epoch %s" % epoch)
            for idx, (image, target) in enumerate(tqdm.tqdm(dataloader, ascii=True)):
                image, target = image.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                predict = self.net(image)
                loss = self.loss(predict, target.squeeze(1))
                loss.backward()
                self.losses.append(loss.item())
                self.optimizer.step()
                if idx % 10 == 0:
                    print(">>> Loss: {}".format(np.mean(self.losses[-10:])))

                if self.config['DEBUG'] == True:
                    break
            print("Trainer epoch finished")
            save_checkpoint(self.net, {"epoch": epoch}, "{}-net.pth".format(epoch))


class Evaluation:
    def __init__(self, net, config):
        net.eval()
        net.to(config['DEVICE'])
        self.device = config['DEVICE']
        self.net = net
        self.config = config
        self.loss = nn.CrossEntropyLoss()

    def run(self, dataloader):
        print(">> Running Evaluation")
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for idx, (image, target) in enumerate(tqdm.tqdm(dataloader, ascii=True)):
                image, target = image.to(self.device), target.to(self.device)
                predict = self.net(image)
                test_loss += self.loss(predict, target.squeeze(1))
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
