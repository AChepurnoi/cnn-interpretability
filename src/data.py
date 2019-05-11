import torch
from torch.utils import data
import numpy as np
from torchvision import datasets, transforms
import pandas as pd
import os
from PIL import Image


class CelebaData(torch.utils.data.Dataset):
    def __init__(self, csv_file, folder="celeba/", image_folder="img_align_celeba"):
        self.img_path = os.path.join(folder, image_folder)
        self.data = pd.read_csv(csv_file)
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        imid, target = self.data.iloc[idx]
        image = Image.open(os.path.join(self.img_path, imid))
        image = self.transforms(image)
        return image, torch.from_numpy(np.array([target]))
