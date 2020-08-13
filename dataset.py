from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
from PIL import Image, ImageEnhance
from fastai.vision import imagenet_stats

from synth_ml.blender.callbacks.metadata import Metadata


class PegInHoleDataset(torch.utils.data.Dataset):
    c = 2
    empty_val = False

    def __init__(self, data_root, overlay_root=None, overlay_filetype='jpg', valid=False, sigma=3,
                 rand_crop=True, flip_horizontal=True, N=1000):
        self.data_root = Path(data_root)
        self.valid = valid
        self.sigma = sigma
        self.rand_crop = rand_crop
        self.overlay_image_fps = list(Path(overlay_root).glob('*.{}'.format(overlay_filetype)))
        self.flip_horizontal = flip_horizontal
        self.N = N
        self.N_VALID = N // 10
        self.N_TRAIN = N - self.N_VALID

    def __len__(self):
        return self.N_VALID if self.valid else self.N_TRAIN

    def get_rep(self, idx):
        data_path = self.data_root / 'metadata' / '{:04}.json'.format(idx)
        metadata = Metadata(data_path)

        w, h = metadata.resolution
        p_hole_img = metadata.world_2_image((0, 0, 0))
        p_peg_img = metadata.world_2_image(
            metadata.objects['Peg'].t_world @ np.array(((0, 0, -1, 1), (0, 0, 0, 1))).T
        )

        heatmaps = np.empty((2, h, w))
        heatmaps[0] = heatmap(self.sigma, w, h, p_hole_img[:2].T)
        heatmaps[1] = heatmap(self.sigma, w, h, p_peg_img[:2, 0:1].T)
        return heatmaps

    def get(self, idx, random_state: np.random.RandomState = np.random):
        if self.valid:
            idx = idx + self.N_TRAIN
        img_path = self.data_root / 'cycles_denoise' / '{:04}.png'.format(idx)
        img = Image.open(str(img_path))
        if self.overlay_image_fps:
            img = overlay_composite(img, self.overlay_image_fps, random_state)[0]
        rep = self.get_rep(idx)
        if self.rand_crop:
            img = np.array(img)
            h, w = img.shape[:2]
            crop_size = 224
            crop_start = random_state.rand(2) * (h - crop_size, w - crop_size)
            h0, w0 = np.round(crop_start).astype(int)
            img = img[h0:h0 + crop_size, w0:w0 + crop_size]
            rep = rep[:, h0:h0 + crop_size, w0:w0 + crop_size]
        if self.flip_horizontal and random_state.rand() < 0.5:
            img = img[:, ::-1].copy()
            rep = rep[:, :, ::-1].copy()
        if random_state.rand() < 0.5:
            k = random_state.randint(1, 3) * 2 + 1
            img = cv2.blur(img, (k, k))
        return img, rep

    @staticmethod
    def normalize(img, rep):
        img = torchvision.transforms.ToTensor()(img)
        img = torchvision.transforms.Normalize(*imagenet_stats)(img)
        rep = torch.from_numpy(rep).float()
        return img, rep

    def __getitem__(self, idx):
        return self.normalize(*self.get(idx))


def heatmap(sigma, w, h, points, d=3):  # efficient version of heatmap naive
    s = int(sigma * d)  # assumes that values further away than sigma * d are insignificant
    hm = np.zeros((h, w))
    for x, y in points:
        _x, _y = int(round(x)), int(round(y))
        xmi, xma = max(0, _x - s), min(w, _x + s)
        ymi, yma = max(0, _y - s), min(h, _y + s)
        _h, _w = yma - ymi, xma - xmi
        X, Y = np.arange(_w).reshape(1, _w), np.arange(_h).reshape(_h, 1)
        _hm = (x - xmi - X) ** 2 + (y - ymi - Y) ** 2
        _hm = np.exp(-_hm / (2 * sigma ** 2))
        hm[ymi:yma, xmi:xma] = np.maximum(hm[ymi:yma, xmi:xma], _hm)
    return hm


def load_rand_overlay(overlay_img_fps, random_state: np.random.RandomState = np.random) -> Image.Image:
    fp = random_state.choice(overlay_img_fps)
    return Image.open(fp)


def overlay_composite(img: Image, overlay_img_fps, random_state: np.random.RandomState = np.random, max_alpha=0.75):
    overlay = load_rand_overlay(overlay_img_fps, random_state).convert('RGB') \
        .resize((img.width, img.height), Image.BILINEAR)
    mask = load_rand_overlay(overlay_img_fps, random_state).convert('L') \
        .resize((img.width, img.height), Image.BILINEAR)
    mask = ImageEnhance.Brightness(mask).enhance(max_alpha)
    return Image.composite(overlay, img, mask), overlay, mask


def main():
    dataset = PegInHoleDataset('synth_ml_data')
    img, rep = dataset.get(0)
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.subplot(1, 3, 2)
    plt.imshow(rep[0])
    # vec_img_math.plot_angle_img(vec_img_math.get_angle_img(vector_fields[0]))
    plt.subplot(1, 3, 3)
    plt.imshow(rep[1])
    # vec_img_math.plot_angle_img(vec_img_math.get_angle_img(vector_fields[1]))
    plt.show()
    print(dataset[0][1].shape)


if __name__ == '__main__':
    main()
