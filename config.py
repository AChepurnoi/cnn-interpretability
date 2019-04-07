import torch

config = {
    'DEBUG': False,
    'CUDA': torch.cuda.is_available(),
    'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu")
}


