import argparse
from brain_stl.main import OPENNEURO_URL, run_brain_stl


def run_cli():
    parser = argparse.ArgumentParser(description='Generate a 3D STL model of the brain from a scan')
    parser.add_argument('-i', '--scan_filepath', type=str, default=None, help='Path to the input scan (NIfTI or DICOM)')
    parser.add_argument('-o', '--output_folder', type=str, default='.', help='Folder to save the STL file and other outputs')
    parser.add_argument('-n', '--stl_name', type=str, default='brain', help='Name of the example STL file (without extension)')
    parser.add_argument('-ni', '--nifti_name', type=str, help='If DICOM produces multiple NifTis, this NifTi filename is used to proceed')
    parser.add_argument('-u', '--url', type=str, default=OPENNEURO_URL, help='URL to download from if scan_filepath is not provided')
    parser.add_argument('-c', '--use_cache', action='store_true', help='Reuses intermediate outputs (e.g., tissue data)')
    parser.add_argument('-th', '--threshold', type=float, default=1.5, help='Threshold value for brain segmentation (0.5:fluid+gray+white matter, 1.5=gray+white matter, 2.5=white matter)')
    parser.add_argument('-hl', '--hollow', type=float, default=0.0, help='How hollow the brain is (0 to 1)')
    parser.add_argument('-s', '--in_template_space', action='store_true', help='If True, the 3D model is in standardized (template) orientation')
    parser.add_argument('-mr', '--mesh_reduction', type=float, default=0.0, help='Fraction of mesh reduction (0 to 1) for smaller STL file size')
    parser.add_argument('-si', '--smooth_iter', type=int, default=0, help='Number of iterations for Laplacian smoothing')
    parser.add_argument('-sl', '--smooth_lamb', type=float, default=0.2, help='Strength of each Laplacian smoothing step')
    parser.add_argument('-t', '--text', type=str, help='Custom text used as structural support')
    parser.add_argument('-t2', '--text2', type=str, help='Second line of text')
    parser.add_argument('-ts', '--textsize', type=int, default=100, help='Size (height) of the text')
    parser.add_argument('-tb', '--textback', type=int, default=300, help='Coronal (depth) position of the back of the text')
    parser.add_argument('-tf', '--textfront', type=int, default=330, help='Coronal (depth) position of the front of the text')
    parser.add_argument('-tl', '--textlinelength', type=int, default=10, help='Length of line below the text')
    args = parser.parse_args()
    run_brain_stl(scan_filepath=args.scan_filepath,
                  output_folder=args.output_folder,
                  stl_name=args.stl_name,
                  use_cache=args.use_cache,
                  hollow=args.hollow,
                  text=args.text,
                  text2=args.text2,
                  textsize=args.textsize,
                  textback=args.textback,
                  textfront=args.textfront,
                  textlinelength=args.textlinelength,
                  threshold=args.threshold,
                  in_template_space=args.in_template_space,
                  mesh_reduction=args.mesh_reduction,
                  smooth_iter=args.smooth_iter,
                  smooth_lamb=args.smooth_lamb,
                  nifti_name=args.nifti_name,
                  url=args.url)


if __name__ == '__main__':
    run_cli()
