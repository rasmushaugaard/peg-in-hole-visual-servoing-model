import enum

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from unet import ResNetUNet


class Model(pl.LightningModule):
    def __init__(self, loss='mse'):
        super().__init__()
        assert loss in ['mse', 'bce']
        self.loss = loss
        self.unet = ResNetUNet(n_class=2)
        self.save_hyperparameters('loss')

    def forward(self, x):
        x = self.unet(x)  # (B, 2, H, W)
        if self.loss == 'bce':
            x = torch.sigmoid(x)
        return x

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-4)

        def lr(i):
            # warmup
            lr_ = min(i / 100, 1.)
            return lr_

        return dict(
            optimizer=opt,
            lr_scheduler=dict(
                scheduler=torch.optim.lr_scheduler.LambdaLR(opt, lr),
                interval='step',
            ),
        )

    def step(self, batch, log_name, log_pbar=False):
        x, y = batch
        lgts = self.unet(x)
        if self.loss == 'mse':
            loss = F.mse_loss(lgts, y)
        elif self.loss == 'bce':
            loss = F.binary_cross_entropy_with_logits(lgts, y)
        else:
            raise ValueError()
        self.log(f'loss_{log_name}', loss, prog_bar=log_pbar)
        return loss

    def training_step(self, batch, _):
        return self.step(batch, 'train')

    def validation_step(self, batch, _):
        return self.step(batch, 'val', True)
