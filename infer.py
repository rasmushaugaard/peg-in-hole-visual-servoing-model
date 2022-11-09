from pathlib import Path
import argparse

import numpy as np
import cv2
import rospy
import sensor_msgs.msg
from ros_numpy.image import image_to_numpy

from dataset import RealDataset
from model import Model

parser = argparse.ArgumentParser()
parser.add_argument('model_path')
parser.add_argument('--img-topic')
args = parser.parse_args()

model = Model.load_from_checkpoint(args.model_path)
model.cuda()
model.eval()
model.freeze()

if args.img_topic is not None:
    rospy.init_node('infer')

real_img_fps = list(Path('real_data/images').glob('*.png'))
dataset = RealDataset(real_img_fps, augs=True)

while True:
    if args.img_topic:
        img = rospy.wait_for_message(args.img_topic, sensor_msgs.msg.Image)
        img = image_to_numpy(img)[..., ::-1]
        h, w = img.shape[:2]
        t, l, res = h // 2, w // 2, 224
        img = img[t:t + res, l:l + res]
        #img = img[t:t + res//2, l:l + res//2]
        #img = cv2.resize(img, (res, res))
    else:
        i = np.random.randint(len(dataset))
        img, _ = dataset.get(i)[:2]
    img_ = dataset.normalize(img)[0][None].to(model.device)
    act = model.forward(img_)[0].cpu().numpy()
    act = np.clip(act, 0, 1)

    comp = img.copy()
    for i, c in enumerate(((0, 0, 255), (255, 0, 0))):
        y, x = np.unravel_index(np.argmax(act[i]), act[i].shape)
        cv2.drawMarker(comp, (x, y), c, cv2.MARKER_CROSS, markerSize=50)

    cv2.imshow('', comp)
    cv2.imshow('hole', act[0])
    cv2.imshow('peg', act[1])

    if cv2.waitKey(1 if args.img_topic else 0) == ord('q'):
        break
