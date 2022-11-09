from typing import Union

import cv2
import numpy as np

COLORS = {
    'r': (255, 0, 0, 255),
    'g': (0, 255, 0, 255),
    'b': (0, 0, 255, 255),
    'k': (0, 0, 0, 255),
    'w': (255, 255, 255, 255),
}


def draw_points(img, points, c: Union[str, tuple] = 'r'):
    if isinstance(c, str):
        c = COLORS[c]
    for i, p in enumerate(points):
        cv2.drawMarker(img, tuple(p[::-1]), c, cv2.MARKER_TILTED_CROSS, 10, 1, cv2.LINE_AA)


def load_rand_overlay(overlay_img_fps, random_state: np.random.RandomState = np.random,
                      size=None, imread=cv2.IMREAD_COLOR):
    fp = random_state.choice(overlay_img_fps)
    img = cv2.imread(str(fp), imread)
    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    return img


def overlay_composite(img, overlay_img_fps, random_state: np.random.RandomState = np.random, max_alpha=0.3):
    h, w = img.shape[:2]
    overlay = load_rand_overlay(overlay_img_fps, random_state, size=(w, h))
    mask = load_rand_overlay(overlay_img_fps, random_state, size=(w, h), imread=cv2.IMREAD_GRAYSCALE)
    mask_f = mask.astype(np.float32)[..., None] * (max_alpha / 255)
    comp = img.astype(np.float32) * (1 - mask_f) + overlay.astype(np.float32) * mask_f
    comp = comp.round().astype(np.uint8)
    return comp, overlay, mask


def heatmap(sigma, w, h, points, d=3):  # efficient version of heatmap naive
    s = int(sigma * d)  # assumes that values further away than sigma * d are insignificant
    hm = np.zeros((h, w))
    for x, y in points:
        _x, _y = int(round(x)), int(round(y))
        xmi, xma = max(0, _x - s), min(w, _x + s)
        ymi, yma = max(0, _y - s), min(h, _y + s)
        _h, _w = yma - ymi, xma - xmi
        if _h > 0 and _w > 0:
            X, Y = np.arange(_w).reshape(1, _w), np.arange(_h).reshape(_h, 1)
            _hm = (x - xmi - X) ** 2 + (y - ymi - Y) ** 2
            _hm = np.exp(-_hm / (2 * sigma ** 2))
            hm[ymi:yma, xmi:xma] = np.maximum(hm[ymi:yma, xmi:xma], _hm)
    return hm
