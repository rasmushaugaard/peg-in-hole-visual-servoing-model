import argparse
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from synth_ml.utils import rand
from synth_ml.blender import bl_utils, materials
from synth_ml.blender.wrappers import Scene, Object
import synth_ml.blender.callbacks as cb

deg = math.pi / 180

parser = argparse.ArgumentParser()
parser.add_argument("--frame-start", type=int, default=0)
parser.add_argument("--frame-end", type=int, default=1000)
args = parser.parse_args(bl_utils.get_user_args())

ts = cb.TransformSampler()

s = Scene.from_name('Scene')
cube = Object.from_name('Cube')
peg = Object.from_name('Peg')
peg_tip = Object.from_name('PegTip')
cam = Object.from_name('Camera')
cam_elevation = Object.from_name('CameraElevation')
hole = Object.from_name('Hole')

material = materials.ProceduralNoiseVoronoiMix()
cube.assign_material(material)
peg.assign_material(material)

ts.pos_axis_uniform(cam, 'y', 0.12, 0.15)
ts.rot_axis_uniform(cam, 'y', - 5 * deg, 5 * deg)
ts.rot_axis_uniform(cam_elevation, 'x', 35 * deg, 45 * deg)


def hyper_sphere_volume_sampler(d, r=1.):
    while True:
        p = np.random.uniform(-r, r, d)
        if np.linalg.norm(p) <= r:
            return p


def hyper_sphere_surface_sampler(d, r=1.):
    p = hyper_sphere_volume_sampler(d)
    return p / np.linalg.norm(p) * r


def peg_pos_sampler():
    xy = hyper_sphere_volume_sampler(2, 0.015)
    z = np.random.uniform(0.005, 0.015)
    return (*xy, z)


def peg_rot_sampler():
    rotvec = hyper_sphere_surface_sampler(3, np.random.uniform(0, 5 * deg))
    return Rotation.from_rotvec(rotvec).as_euler('xyz')


ts.add_pos_sampler(peg_tip, peg_pos_sampler)
ts.add_rot_sampler(peg_tip, peg_rot_sampler)

folder = 'synth_data'
Path(folder).mkdir(exist_ok=True)
s.callback = cb.CallbackCompose(
    ts,
    material.sampler_cb(
        scale=100, p_metallic=0.5,
        roughness=rand.NormalFloatSampler(0.1, .5, mi=0, ma=1),
        bevel_radius=rand.UniformFloatSampler(0.0005, 0.002)
    ),
    cb.HdriEnvironment(scene=s, category='all', resolution=1, max_altitude_deg=90),
    cb.MetadataLogger(scene=s, objects=[peg], output_folder=folder),
    cb.Renderer(
        output_folder=folder,
        scene=s, resolution=280, engines=('CYCLES',), denoise=True, save_image=False,
        cycles_samples=rand.NormalFloatSampler(mu=0, std=50, mi=5, ma=64, round=True),
    ),
)

s.render_frames(range(args.frame_start, args.frame_end))
