import json
import enum
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('image_folder')
parser.add_argument('--filetype', default='png')
parser.add_argument('--img-scale', type=float, default=1.)
parser.add_argument('--roi-size', type=int, default=1200)

args = parser.parse_args()
img_scale = args.img_scale
roi_size = args.roi_size

M = cv2.getRotationMatrix2D((0, 0), 0, img_scale)

data_folder = Path(args.image_folder)
image_folder = data_folder / 'images'
annotation_folder = data_folder / 'annotations'
annotation_folder.mkdir(exist_ok=True)

image_names = set([f.name[:-4] for f in image_folder.glob(f'*.{args.filetype}')])
print(len(image_names))
annotation_names = set([f.name[:-5] for f in annotation_folder.glob('*.json')])
names_to_be_annotated = image_names - annotation_names

i = 0
for name in tqdm(names_to_be_annotated):
    image_fp = image_folder / f'{name}.{args.filetype}'
    img = cv2.imread(str(image_fp))
    h, w = (np.array(img.shape[:2]) * img_scale).astype(int)
    img_scaled = cv2.warpAffine(img, M, (w, h))
    cv2.imshow('', img_scaled)


    class State(enum.Enum):
        ROI = 0
        HOLE_SIZE = 1
        HOLE_DIRECTION = 2
        PEG_POSITION = 3
        CONFIRM = 4


    state = State.ROI
    start = hole_size = hole_position = hole_direction = peg_position = roi = img_roi = roi_scale = None


    def cb(event, x, y, flags, _):
        global state, start, hole_size, hole_position, hole_direction, peg_position, roi, img_roi, roi_scale
        if state == State.ROI:
            if event == cv2.EVENT_LBUTTONDOWN:
                start = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                l, t, r, b = *start, x, y
                l, r = min(l, r), max(l, r)
                t, b = min(t, b), max(t, b)
                l, t, r, b = (np.array((l, t, r, b)) / img_scale).round().astype(int)
                roi = l, t, r, b
                roi_scale = roi_size / max(r - l, b - t)
                M_roi = cv2.getRotationMatrix2D((0, 0), 0, roi_scale)
                img_roi = img[t:b + 1, l:r + 1]
                img_roi = cv2.warpAffine(img_roi, M_roi, (roi_size, roi_size))
                cv2.imshow('', img_roi)
                start = None
                state = State.HOLE_SIZE
            if start is not None:
                img_ = img_scaled.copy()
                cv2.rectangle(img_, start, (x, y), (255, 0, 0), lineType=cv2.LINE_AA)
                cv2.imshow('', img_)
        elif state == State.HOLE_SIZE:
            if event == cv2.EVENT_LBUTTONDOWN:
                start = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                state = State.HOLE_DIRECTION
                hole_size = np.linalg.norm(np.array((x, y)) - start)
                hole_position = (x + start[0]) // 2, (y + start[1]) // 2
            if start is not None:
                img_ = img_roi.copy()
                cv2.line(img_, start, (x, y), (255, 0, 0), lineType=cv2.LINE_AA)
                cv2.imshow('', img_)
        elif state == State.HOLE_DIRECTION:
            img_ = img_roi.copy()
            cv2.line(img_, hole_position, (x, y), (0, 255, 0), lineType=cv2.LINE_AA)
            cv2.imshow('', img_)
            if event == cv2.EVENT_LBUTTONUP:
                hole_direction = np.arctan2(y - hole_position[1], x - hole_position[0])
                state = State.PEG_POSITION
        elif state == State.PEG_POSITION:
            img_ = img_roi.copy()
            cv2.circle(img_, (x, y), round(hole_size / 2), (0, 0, 255), lineType=cv2.LINE_AA)
            cv2.imshow('', img_)
            if event == cv2.EVENT_LBUTTONUP:
                peg_position = (x, y)

                # show final image
                img_ = img.copy()
                roi_lt = np.array(roi[:2])

                hole_size = hole_size / roi_scale
                hole_position = roi_lt + np.array(hole_position) / roi_scale
                peg_position = roi_lt + np.array(peg_position) / roi_scale

                for pos, col in (hole_position, (255, 0, 0)), (peg_position, (0, 0, 255)):
                    pos = pos.round().astype(int)
                    thickness = round(1 / img_scale)
                    radius = round(hole_size / 2)
                    cv2.circle(img_, pos, radius, col, thickness, cv2.LINE_AA)
                    cv2.drawMarker(img_, pos, col, cv2.MARKER_CROSS, radius, thickness, cv2.LINE_AA)

                img_ = cv2.warpAffine(img_, M, (w, h))
                cv2.imshow('', img_)

                state = State.CONFIRM


    cv2.setMouseCallback('', cb)
    while True:
        key = cv2.waitKey()
        if key == ord('q'):
            quit()
        elif key in (ord('\r'), ord(' ')) and state == State.CONFIRM:
            annotation = dict(
                hole_size=hole_size,
                hole_position=hole_position.tolist(),
                peg_position=peg_position.tolist(),
                hole_direction=hole_direction,
            )
            annotation_fp = annotation_folder / f'{image_fp.name[:-4]}.json'
            json.dump(annotation, annotation_fp.open('w'), indent=2)
            break
        elif key == ord('s'):
            break
