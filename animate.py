"""Render brain.stl models as rotating, animated AVIFs with a pyrender-like look (no OpenGL needed)."""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import uniform_filter

from brain_stl.main import OPENNEURO_URL


def save_brain_avif(stl_filepath, output_folder='.', name='brain', fps=33, quality=50, **kwargs):
    """Animated AVIF of a single model, cropped as tightly as possible."""
    save_avif([trimesh.load_mesh(stl_filepath)], f'{output_folder}/{name}.avif', fps, quality, **kwargs)


def save_readme_avif(output_folder='.', name='brain', fps=33, quality=50, **kwargs):
    """README animation: white matter only (threshold 2.5), default settings, and with 'YOUR BRAIN' text side by side."""
    from brain_stl import run_brain_stl
    scan = Path(output_folder) / OPENNEURO_URL.split(':')[-1]
    kw = dict(scan_filepath=str(scan) if scan.is_file() else None,  # reuse the scan instead of downloading it 3 times
              output_folder=output_folder, use_cache=True, stl_name=None, in_template_space=True)
    meshes = [run_brain_stl(**kw, threshold=2.5), run_brain_stl(**kw), run_brain_stl(**kw, text='YOUR', text2='BRAIN')]
    save_avif(meshes, f'{output_folder}/{name}.avif', fps, quality, **kwargs)


def save_avif(meshes, avif_filepath, fps=33, quality=50, **kwargs):
    """Render the meshes side by side (one process each) and crop to the tightest box containing all content."""
    kwargs = {'zoom': .53, **kwargs}  # .53 = scale of the original gif (a 640x480 render, downscaled)
    with ProcessPoolExecutor(len(meshes)) as pool:
        tiles = list(pool.map(partial(render_frames, **kwargs), meshes))
    boxes = np.array([frame.getbbox() for tile in tiles for frame in tile])  # same box for all frames: no jitter
    box = (*boxes[:, :2].min(0) - 1, *boxes[:, 2:].max(0) + 1)
    tiles = [[frame.crop(box) for frame in tile] for tile in tiles]
    width, height = tiles[0][0].size
    frames = []
    for parts in zip(*tiles):
        frame = Image.new('L', (width * len(parts), height))
        for i, part in enumerate(parts):
            frame.paste(part, (i * width, 0))
        frames.append(frame.convert('RGB'))
    frames[0].save(avif_filepath, save_all=True, append_images=frames[1:], duration=round(1000 / fps), loop=0,
                   quality=quality)


def render_frames(mesh, n=180, size=(640, 480), zoom=1., radius=190, fov=60, light=(80, -190, 80), peak=.57,
                  sharpness=1.1, ambient=.07, wrap=0., occlusion=3., flat=True):
    """Point-splat renderer using a pyrender-style camera and point light: the camera looks along +y from
    y=-radius, the light has 1/d^2 falloff. The mesh spins around z, starting at 180 degrees.

    Shading is `peak * tanh(sharpness * (ambient + lambert))`, so highlights saturate softly. `wrap` > 0 lets light
    wrap around the terminator (softer, weaker shadows). `occlusion` is the depth (in mm) a point may lie behind the
    nearest surface before it is hidden. `flat` splats triangle centres with face normals (faceted, grainy shading)
    instead of vertices with vertex normals (smoother).
    """
    w, h = size
    points, normals = (mesh.triangles_center, mesh.face_normals) if flat else (mesh.vertices, mesh.vertex_normals)
    points = (points - mesh.vertices.mean(0)).astype(np.float32)
    normals = normals.astype(np.float32)
    camera = np.array([0, -radius, 0], dtype=np.float32)
    light = np.array(light, dtype=np.float32)
    focal = zoom * (h / 2) / np.tan(np.radians(fov) / 2)
    frames = []
    for angle in np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi:
        rot = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])[:3, :3].T.astype(np.float32)
        pos, nrm = points @ rot, normals @ rot
        front = ((camera - pos) * nrm).sum(1) > 0  # cull back-facing points
        pos, nrm = pos[front], nrm[front]
        depth = pos[:, 1] + radius
        px = np.floor(w / 2 + focal * pos[:, 0] / depth).astype(np.int32)
        py = np.floor(h / 2 - focal * pos[:, 2] / depth).astype(np.int32)
        to_light = light - pos
        dist = np.linalg.norm(to_light, axis=1)
        lambert = np.clip(((nrm * to_light).sum(1) / dist + wrap) / (1 + wrap), 0, 1)
        shade = 255 * peak * np.tanh(sharpness * (ambient + lambert) * (radius / dist) ** 2)
        inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        buf = splat(px[inside], py[inside], depth[inside], shade[inside].astype(np.uint8), size, occlusion)
        frames.append(Image.fromarray(close_holes(buf)))
    return frames


def splat(px, py, depth, shade, size, occlusion):
    """Z-buffer: draw the nearest point per pixel, after dropping points that lie clearly behind the nearest surface
    in their 3x3 neighbourhood (sulcal walls, far side) because those would shine through the gaps between points."""
    w, h = size
    zbuf = np.full((h, w), np.inf, np.float32)
    far_to_near = np.argsort(-depth)  # nearest write wins
    zbuf[py[far_to_near], px[far_to_near]] = depth[far_to_near]
    zmin = np.min([np.roll(zbuf, (dy, dx), axis=(0, 1)) for dy in (-1, 0, 1) for dx in (-1, 0, 1)], axis=0)
    keep = far_to_near[depth[far_to_near] <= zmin[py[far_to_near], px[far_to_near]] + occlusion]
    buf = np.zeros((h, w), np.uint8)
    buf[py[keep], px[keep]] = shade[keep]
    return buf


def close_holes(buf, reach=6):
    """Fill empty pixels that have surface on two opposite sides within `reach` px, so the silhouette stays crisp."""
    lit = buf > 0
    enclosed = np.zeros_like(lit)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        before = np.any([np.roll(lit, (d * dy, d * dx), axis=(0, 1)) for d in range(1, reach + 1)], axis=0)
        after = np.any([np.roll(lit, (-d * dy, -d * dx), axis=(0, 1)) for d in range(1, reach + 1)], axis=0)
        enclosed |= before & after
    window = 2 * reach + 1
    local_mean = uniform_filter(buf.astype(np.float32), window) / np.maximum(uniform_filter(lit.astype(np.float32), window), 1e-6)
    return np.where(~lit & enclosed, local_mean.astype(np.uint8), buf)


if __name__ == '__main__':
    save_readme_avif()
