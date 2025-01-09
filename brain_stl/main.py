import trimesh
import dcm2niix
import numpy as np
import pandas as pd
import nibabel as nib
import urllib.request
import fast_simplification
from pathlib import Path
from mcubes import marching_cubes
from deepbet.utils import dilate
from niftiview import NiftiImageGrid
from PIL import Image, ImageDraw, ImageFont
from deepmriprep.preprocess import AFFINE_TEMPLATE, Preprocess
OPENNEURO_URL = 'https://openneuro.org/crn/datasets/ds000001/snapshots/1.0.0/files/sub-01:anat:sub-01_T1w.nii.gz'
OUTPUTS = ['affine', 'rotation', 'p0_large']


def run_brain_stl(scan_filepath=None, output_folder='.', stl_name='brain', nifti_name=None, url=OPENNEURO_URL,
                  use_cache=False, threshold=1.5, hollow=.0, in_template_space=False, mesh_reduction=.0, smooth_iter=0,
                  smooth_lamb=.2, text=None, text2=None, textsize=120, textback=300, textfront=330, textlinelength=10):
    assert Path(output_folder).is_dir(), f'Output folder {output_folder} does not exist'
    if scan_filepath is None:
        print(f'Instead of a scan from disk, using a scan downloaded from {url}')
        nifti_filepath = f'{output_folder}/{url.split(":")[-1]}'
        urllib.request.urlretrieve(url, nifti_filepath)
    else:
        assert Path(str(scan_filepath)).exists(), f'Scan filepath {scan_filepath} does not exist'
        if Path(scan_filepath).is_file() and not scan_filepath.endswith('.dcm'):
            nifti_filepath = scan_filepath
        else:
            nifti_filepath = process_dicom(scan_filepath, output_folder, nifti_name)
    output_paths = {out: f'{output_folder}/{"tissue.nii.gz" if out == "p0_large" else out+".csv"}' for out in OUTPUTS}
    if not use_cache or not all([Path(output_paths[output]).is_file() for output in OUTPUTS]):
        print(f'Calculate and cache (=write to output folder) affine, rotation and tissue of {nifti_filepath}')
        prep = Preprocess()
        prep.run(nifti_filepath, output_paths=output_paths, run_all=False)
    tissue_img = nib.as_closest_canonical(nib.load(output_paths['p0_large']))
    array = tissue_img.get_fdata(dtype=np.float32)
    array = array if hollow == 0 else hollow_out_brain(array, hollow)
    array = array if text is None else add_text(array, text, text2, textsize, textback, textfront, textlinelength)
    vertices, faces = marching_cubes(array, threshold)
    if in_template_space:
        vertices = to_template_space(vertices)
    else:
        vertices = to_scanner_space(vertices, array, nifti_filepath, affine_filepath=output_paths['affine'])
    if mesh_reduction > 0:
        vertices, faces = fast_simplification.simplify(vertices.tolist(), faces.tolist(), mesh_reduction)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    if smooth_iter > 0:
        print('Smoothing')
        mesh = trimesh.smoothing.filter_laplacian(mesh, lamb=smooth_lamb, iterations=smooth_iter)
    if stl_name is not None:
        mesh.export(f'{output_folder}/{stl_name}.stl')
    return mesh


def process_dicom(scan_filepath, output_folder, nifti_name):
    dcm2niix.main([f'-o', output_folder, scan_filepath])
    nifti_files = sorted(Path.glob(f'{output_folder}/*.ni*'))
    assert len(nifti_files) > 0, f'Scan filepath {scan_filepath} is no valid DICOM'
    if len(nifti_files) > 1:
        im = get_t1w_png(nifti_files)
        im.save(f'{output_folder}/pick_t1w_image.png')
        raise RuntimeError(f'The inputted DICOM {scan_filepath} resulted in {len(nifti_files)} Nifti files.'
                           f'View {output_folder}/pick_t1w_image.png, select the filename'
                           f'of the appropriate(=T1w) Nifti and pass it via "run_brain_stl(..., nifti_name=*)"')
    return nifti_files[0] if nifti_name is None else [fp for fp in nifti_files if nifti_name in fp][0]


def hollow_out_brain(array, hollow):
    r = int(.3 * (array.shape[0] * array.shape[1] * array.shape[2]) ** (1 / 3))
    hollow_mask = dilate(array > 0, n_layer=int((hollow - 1) * r))
    array[hollow_mask] = 0
    return array


def to_template_space(vertices):
    vertices = AFFINE_TEMPLATE.affine @ np.hstack([vertices, np.ones_like(vertices[:, :1])]).T
    return vertices[:3].T


def to_scanner_space(vertices, array, nifti_filepath, affine_filepath):
    affine = pd.read_csv(affine_filepath).values
    affine = affine[[2, 1, 0, 3]][:, [2, 1, 0, 3]]  # undo pytorch zyx ordering
    img = nib.as_closest_canonical(nib.load(nifti_filepath))
    vertices = (2 * vertices / np.array(array.shape)) - 1  # apply pytorch grid scaling [-1;1]
    vertices = (affine @ np.hstack([vertices, np.ones_like(vertices[:, :1])]).T)[:3]
    vertices = .5 * np.array(img.get_fdata().shape) * (vertices.T + 1)  # undo pytorch grid scaling [-1;1]
    vertices = img.affine @ np.hstack([vertices, np.ones_like(vertices[:, :1])]).T
    return vertices[:3].T


def add_text(array, text, text2, size, back=300, front=330, linelength=10):
    size = size if text2 is None else size // 2
    textrange = range(back, front)
    height = np.where(array[:, textrange].mean(axis=(0, 1)) > .2)[0][0]
    text_array = get_text_array(text, array.shape[0], size)
    text_left, text_right, text_low, text_up = get_text_bbox(text_array, linelength)
    fontheight = text_up - text_low
    array[:, textrange, height - fontheight:height] += text_array[:, None, text_low:text_up]
    if text2 is None:
        array[text_left:text_right, textrange, 1:height - fontheight + 2] += 3
    else:
        text_array = get_text_array(text2, array.shape[0], size)
        text_left, text_right, text_low, text_up = get_text_bbox(text_array, linelength)
        fontheight2 = text_up - text_low
        array[:, textrange, 1:fontheight2+1] = text_array[:, None, text_low:text_up]
        array[text_left:text_right, textrange, fontheight2+1:height-fontheight+1] += 3
        array[text_left:text_right, textrange, 1:2] += 3
    return array.clip(max=3)


def get_text_array(text, width, height):
    im = Image.new('L', (width, height), color=0)
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default(height)
    d.text((width // 2, 0), text, fill=3, font=font, anchor='mt')
    return np.array(im).T[::-1, ::-1]


def get_text_bbox(array, linelength):
    text_left, text_right = np.where(array.sum(axis=1))[0][[0, -1]]
    text_left = max(0, text_left - linelength - 1)
    text_right = min(text_right + linelength - 1, array.shape[0])
    text_low, text_up = np.where(array.sum(axis=0))[0][[0, -1]]
    return text_left, text_right, text_low, text_up


def get_t1w_png(nifti_files):
    ref_im = Image.open(Path(__file__).parent.resolve() / 'data' / 'ref.png')
    grid_im = NiftiImageGrid(nifti_files).get_image(height=800, layout='sagittal++', fpath=1, fontsize=10)
    ref_im = ref_im.resize((grid_im.size[0], int(ref_im.size[1] * grid_im.size[0] / ref_im.size[0])))
    im = Image.new('L', (ref_im.size[0], ref_im.size[1] + grid_im.size[1]))
    im.paste(ref_im, (0, 0))
    im.paste(grid_im, (0, ref_im.size[1]))
    return im
