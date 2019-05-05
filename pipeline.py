import argparse

from src.data import CelebaData
from src.train import Trainer, Evaluation
from src.utils import load_checkpoint
from src.config import config
from src.net import VggNetwork
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", help="Number of epochs to train", type=int, default=1)
parser.add_argument("--batch", help="Number of epochs to train", type=int, default=1)
parser.add_argument("--workers", help="Checkpoint name", type=int, default=6)
parser.add_argument("--checkpoint", help="Checkpoint name", type=str, default=None)

args = parser.parse_args()
print("Arguments: {}".format(args))
print("Config: {}".format(config))

net = VggNetwork(out=2)

if args.checkpoint is not None:
    extra = load_checkpoint(net, args.checkpoint)

train_data = CelebaData("data/train.csv")
val_data = CelebaData("data/val.csv")

train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch, num_workers=args.workers, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch, num_workers=args.workers, shuffle=True)

trainer = Trainer(net, config)
eval = Evaluation(net, config)

for i in range(args.epochs):
    trainer.run(train_loader, epochs=1)
    result = eval.run(val_loader)
    print(f"Validation result: {result['loss']} / {result['accuracy'] * 100}")
