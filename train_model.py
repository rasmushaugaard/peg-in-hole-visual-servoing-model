import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
import fastai.vision as faiv

from dataset import PegInHoleDataset
from unet import ResNetUNet

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--n-epochs', type=int, default=15)
parser.add_argument('--dataset_size', type=int, default=1000)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

data_path = 'synth_data'
overlay_path = '/data/coco/images/val2017'

data_train = PegInHoleDataset(data_path, overlay_path, N=args.dataset_size)
data_valid = PegInHoleDataset(data_path, overlay_path, N=args.dataset_size, valid=True)

if args.debug:
    for i in range(10):
        img, rep = data_train.get(np.random.randint(0, 30))
        plt.imshow(img)
        plt.show()
    quit()

data = faiv.DataBunch.create(data_train, data_valid, bs=16, worker_init_fn=lambda *_: np.random.seed())
model = ResNetUNet(data_train.c)
learner = faiv.learner.Learner(data, model, loss_func=torch.nn.MSELoss()).to_fp16()

learner.fit_one_cycle(args.n_epochs, args.lr, wd=args.weight_decay)
learner.save('model')
