import torch
from torch.utils import data
import numpy as np
from torchvision import datasets, transforms
import pandas as pd
import os
from PIL import Image


class CelebaData(torch.utils.data.Dataset):
    def __init__(self, folder="celeba/", attr_csv="list_attr_celeba.csv",
                 image_folder="img", feature="Smiling", samples=2000):
        self.attrs = pd.read_csv(os.path.join(folder, attr_csv))
        self.img_path = os.path.join(folder, image_folder)
        idx = int(np.random.rand() * 25000)
        negs = self.attrs[self.attrs[feature] == -1].iloc[idx:idx + samples]
        pos = self.attrs[self.attrs[feature] == 1].iloc[idx:idx + samples]
        self.data = pd.concat([negs, pos])
        self.image_ids = self.data['image_id']
        self.target = (self.data[feature] + 1) / 2
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        imid = self.image_ids.iloc[idx]
        target = self.target.iloc[idx]
        image = Image.open(os.path.join(self.img_path, imid))
        image = self.transforms(image)
        return image, torch.FloatTensor(np.array([target]))
