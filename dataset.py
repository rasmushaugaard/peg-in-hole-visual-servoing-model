import json
from pathlib import Path

import cv2
import torch
import numpy as np
from synth_ml.blender.callbacks.metadata import Metadata
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transform3d import Transform

import utils

_augs = [
    A.ColorJitter(brightness=0.4, contrast=0.4),
    A.ISONoise(),
    A.GaussNoise(),
    A.GaussianBlur(blur_limit=(3, 21)),
]


class SynthDataset(torch.utils.data.Dataset):
    def __init__(self, image_fps, overlay_image_fps=None, sigma=5, augs=False):
        self.image_fps = image_fps
        self.sigma = sigma
        self.overlay_image_fps = overlay_image_fps
        self.augs = augs

    def __len__(self):
        return len(self.image_fps)

    def get_annotation(self, idx):
        img_fp = self.image_fps[idx]
        meta_fp = img_fp.parent.parent / 'metadata' / f'{img_fp.name[:-4]}.json'
        metadata = Metadata(meta_fp)
        hole_pos = metadata.world_2_image((0, 0, 0))[:2, 0]
        peg_pos = metadata.world_2_image(metadata.objects['Peg'].t_world @ (0, 0, -1, 1))[:2, 0]
        hole_size = 50.
        hole_direction = np.pi / 2
        return peg_pos, hole_pos, hole_size, hole_direction

    def get_rep(self, peg_pos, hole_pos, res):
        heatmaps = np.empty((2, res, res), dtype=np.float32)
        heatmaps[0] = utils.heatmap(self.sigma, res, res, [hole_pos])
        heatmaps[1] = utils.heatmap(self.sigma, res, res, [peg_pos])
        return heatmaps

    def get(self, idx, random_state: np.random.RandomState = np.random):
        img_path = self.image_fps[idx]
        img = cv2.imread(str(img_path))
        assert img is not None
        peg_pos, hole_pos, hole_size, hole_direction = self.get_annotation(idx)

        res = 224
        # place hole in center along x, 1/3 down along y
        scale = 50. / hole_size
        angle = np.pi / 2 - hole_direction
        cx, cy = res / 2, res / 3

        if self.augs:
            scale *= random_state.uniform(0.6, 1.1)
            cx, cy = random_state.uniform(-25, 25, 2) + (cx, cy)
            angle += random_state.uniform(-np.deg2rad(25), np.deg2rad(25))

        M = (
                    Transform(p=(cx, cy, 0)).matrix @
                    Transform(rotvec=(0, 0, angle)).matrix @
                    (np.eye(4) * (scale, scale, 1, 1)) @
                    Transform(p=(*(-hole_pos), 0)).matrix
            )[:2, (0, 1, 3)]
        img = cv2.warpAffine(img, M, (res, res), borderMode=cv2.BORDER_REFLECT)

        if self.overlay_image_fps and random_state.rand() < 0.5:
            img = utils.overlay_composite(img, self.overlay_image_fps, random_state)[0]
        if self.augs:
            a = _augs.copy()
            random_state.shuffle(a)
            a = A.Compose(a)
            img = a(image=img)['image']

        peg_pos, hole_pos = M @ (*peg_pos, 1), M @ (*hole_pos, 1)
        rep = self.get_rep(peg_pos, hole_pos, res)

        if self.augs and random_state.rand() < 0.5:
            img = img[:, ::-1].copy()
            rep = rep[:, :, ::-1].copy()
            peg_pos[0] = res - peg_pos[0] - 1
            hole_pos[0] = res - hole_pos[0] - 1

        return img, rep, peg_pos, hole_pos

    @staticmethod
    def normalize(img, rep=None):
        img = A.Compose([
            A.Normalize(),
            ToTensorV2(),
        ])(image=img)['image']
        if rep is not None:
            rep = torch.from_numpy(rep)
        return img, rep

    def __getitem__(self, idx):
        return self.normalize(*self.get(idx)[:2])


class RealDataset(SynthDataset):
    def get_annotation(self, idx):
        img_fp = self.image_fps[idx]
        annotation_fp = img_fp.parent.parent / 'annotations' / f'{img_fp.name[:-4]}.json'
        anno = json.load(annotation_fp.open())
        hole_pos = np.array(anno['hole_position'])
        peg_pos = np.array(anno['peg_position'])
        hole_size = anno['hole_size']
        hole_direction = anno['hole_direction']
        return peg_pos, hole_pos, hole_size, hole_direction


def main():
    overlay_img_fps = list(Path('/data/coco/images/val2017').glob('*.jpg'))
    data_kwargs = dict(augs=True, overlay_image_fps=overlay_img_fps)
    data_kwargs = dict(augs=True)
    if False:
        image_fps = list(Path('synth_data/cycles_denoise').glob('*.png'))
        dataset = SynthDataset(image_fps, **data_kwargs)
    else:
        image_fps = list(Path('real_data/images').glob('*.png'))
        dataset = RealDataset(image_fps, **data_kwargs)

    while True:
        img, rep, peg_pos, hole_pos = dataset.get(np.random.randint(len(dataset)))
        for p, c in (peg_pos, (0, 0, 255)), (hole_pos, (255, 0, 0)):
            cv2.drawMarker(img, p.round().astype(int), c, cv2.MARKER_TILTED_CROSS, 10)
        cv2.imshow('hole', rep[0])
        cv2.imshow('peg', rep[1])
        cv2.imshow('', img)
        if cv2.waitKey() == ord('q'):
            break


if __name__ == '__main__':
    main()
