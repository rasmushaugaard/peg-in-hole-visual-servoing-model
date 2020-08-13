from typing import Union

import cv2

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
