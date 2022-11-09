import argparse
from pathlib import Path

import numpy as np
import torch.utils.data
import pytorch_lightning as pl

from dataset import SynthDataset, RealDataset
from model import Model

parser = argparse.ArgumentParser()
parser.add_argument('--loss', choices=['mse', 'bce'], default='mse')
args = parser.parse_args()

data_path = 'synth_data'
overlay_img_fps = list(Path('/data/coco/images/val2017').glob('*.jpg'))

real_img_phone_fps = list(Path('real_data_phone/images').glob('*.jpg'))
real_img_fps = list(Path('real_data/images').glob('*.png'))
synth_img_fps = list(Path('synth_data/cycles_denoise').glob('*.png'))
for fps in real_img_phone_fps, real_img_fps, synth_img_fps:
    assert len(fps) > 0

np.random.shuffle(real_img_phone_fps)
np.random.shuffle(real_img_fps)
train_kwargs = dict(augs=True, overlay_image_fps=overlay_img_fps)
n_valid = 4
data_train = SynthDataset(synth_img_fps, **train_kwargs) + \
             RealDataset(real_img_phone_fps[n_valid:], **train_kwargs) + \
             RealDataset(real_img_fps[n_valid:], **train_kwargs)
data_valid = RealDataset(real_img_phone_fps[:n_valid]) + \
             RealDataset(real_img_fps[:n_valid])

loader_kwargs = dict(batch_size=8, num_workers=5, persistent_workers=True)
loader_train = torch.utils.data.DataLoader(dataset=data_train, shuffle=True, drop_last=True, **loader_kwargs)
loader_valid = torch.utils.data.DataLoader(dataset=data_valid, **loader_kwargs)

model = Model(loss=args.loss)
learner = pl.Trainer(
    gpus=[0], callbacks=[
        pl.callbacks.ModelCheckpoint(monitor='loss_val', save_weights_only=True),
        pl.callbacks.EarlyStopping(monitor='loss_val', patience=10)
    ]
)

learner.fit(model, train_dataloaders=loader_train, val_dataloaders=loader_valid)
