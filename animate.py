import trimesh
import pyrender
import subprocess
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
# To create higher quality GIFs, install gifski and set GIFSKI_PATH
GIFSKI_PATH = None  # Replace with gifski-path (run "which gifski" in terminal) e.g. '/home/USER/.cargo/bin/gifski'


def save_brain_gif(stl_filepath, output_folder='.', gif_name='brain', fps=33, quality=90, **kwargs):
    gif_filepath = f'{output_folder}/{gif_name}.gif'
    mesh = trimesh.load_mesh(stl_filepath)
    frames = get_gif_frames(mesh, **kwargs)
    if GIFSKI_PATH is None:
        frames[0].save(gif_filepath, append_images=frames[1:], save_all=True, duration=int(1000 / fps), loop=0)
    else:
        frames_folder = f'{output_folder}/frames'
        if Path(frames_folder).exists():
            for file in Path(frames_folder).iterdir():
                if file.is_file():
                    file.unlink()
        else:
            Path(frames_folder).mkdir(parents=True, exist_ok=True)
        save_gifski(frames, gif_filepath, frames_folder, fps, quality)


def get_gif_frames(mesh, n=180, size=(640, 480), radius=190, fov=60, light=8, light_angle=(80, 80), bg_color=(0, 0, 0)):
    camera_pose = np.array([[-1, 0, 0, 0], [0, 0, -1, -radius], [0, 1, 0, 0], [0, 0, 0, 1]])
    light_pose = np.array([[1, 0, 0, light_angle[0]], [0, 1, 0, -radius], [0, 0, 1, light_angle[1]], [0, 0, 0, 1]])
    scene = pyrender.Scene(bg_color=bg_color)
    camera = pyrender.PerspectiveCamera(yfov=fov * np.pi / 180)
    light = pyrender.PointLight(color=np.ones(3), intensity=light * 10000)
    renderer = pyrender.OffscreenRenderer(viewport_width=size[0], viewport_height=size[1])
    scene.add(camera, pose=camera_pose)
    scene.add(light, pose=light_pose)
    frames = []
    for angle in tqdm(range(0, 360, 360 // n)):
        pose = trimesh.transformations.rotation_matrix((angle + 180) * np.pi / 180, direction=[0, 0, 1])
        scene.add(pyrender.Mesh.from_trimesh(mesh), pose=pose)
        color, depth = renderer.render(scene)
        frames.append(Image.fromarray(color))
        scene.remove_node(list(scene.mesh_nodes)[0])
    return frames


def save_gifski(frames, gif_filepath, frames_folder, fps=33, quality=90):
    for i, frame in enumerate(frames):
        frame.save(f'{frames_folder}/frame{i}.png')
    subprocess.run(f'{GIFSKI_PATH} --fps {fps} -Q {quality} -o {gif_filepath} {frames_folder}/*.png',
                   stdout=subprocess.PIPE, universal_newlines=True, shell=True)


if __name__ == '__main__':
    from brain_stl import run_brain_stl

    run_brain_stl(in_template_space=False)
    save_brain_gif('brain.stl')
