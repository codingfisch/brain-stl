# brain-stl 🧠
The **easiest** and **fastest** way to turn brain MRI scans into **3D printable models** 🖨️ powered by [`deepmriprep`](https://github.com/wwu-mmll/deepmriprep) ✨

[Use the Colab demo](https://colab.research.google.com/drive/10WSG_qmKnbYrFTwZOlXsIgbda8iz2duQ?usp=sharing) or **install** it locally via `pip install brain_stl` 💨

By default, the model is saved as `brain.stl`, but you can customize the filename along with other options like
- **Threshold**: Pick a value between 0 and 3 for different tissue types (see GIF left, `2.5`: white matter only) 🧠
- **Hollowness**: Adjust how hollow the brain is (enables faster printing) 💨
- **Text**: Add personalized text that works as a support structure 🔤
- **Smoothing**: Achieve a polished surface via iterative Laplacian smoothing ✨
- **Mesh Reduction**: Simplify the mesh to reduce the size of `brain.stl` ✂️

![gif](https://github.com/user-attachments/assets/64f10b4f-b136-4e86-8ef4-37f9fe0d8fa7)

## Usage 💻
To get your first `brain.stl` just run
```python
from brain_stl import run_brain_stl

run_brain_stl()  # per default output_folder='.' (current folder)
```
and a publicly available brain MRI scan (from [OpenNeuro](https://openneuro.org/)) will be used 🌐

Scan files can be inputted in NifTi (`.nii` or `.nii.gz`) or DICOM (`.dcm` or  [folder](https://pydicom.github.io/pydicom/stable/tutorials/filesets.html)) format
```python
from brain_stl import run_brain_stl

run_brain_stl('/path/to/scan.nii.gz')  # or '/path/to/dicom_folder'
```
The main arguments of `run_brain_stl` are
- **`scan_filepath`**: Path to the input scan (if `None`, use scan from `url`). Default: `None`
- **`output_folder`**: Folder to save the `.stl` file and other outputs. Default: `'.'`
- **`stl_name`**: Name of the example `.stl` file (without extension). Default: `'brain'`

<details>
  <summary><b>Click here</b>, to see <b>all</b> remaining <b>arguments</b> 📑</summary>

- **`nifti_name`**: If DICOM produces multiple NifTis, this NifTi filename is used to proceed. Default: `None`
- **`url`**: URL to download a brain scan if no `scan_filepath` is provided. Default: `OPENNEURO_URL`
- **`use_cache`**: If `True`, reuses intermediate outputs (e.g., tissue data). Default: `False`
- **`threshold`**: Threshold tissue value (`0.5`: fluid, `1.5`: gray matter, `2.5`: white matter). Default: `1.5`
- **`hollow`**: How hollow the brain is (0 to 1). Default: `0`
- **`in_template_space`**: If `True`, the 3D model is in standardized (template) orientation. Default: `False`
- **`mesh_reduction`**: Fraction of mesh reduction (0 to 1) for smaller `.stl` file size. Default: `0` (no reduction)
- **`smooth_iter`**: Number of iterations for Laplacian smoothing. Default: `0` (none)
- **`smooth_lamb`**: Strength of each Laplacian smoothing step. Default: `0.2`
- **`text`**: Custom text used as structural support. Default: `None`
  - **`text2`**: Second line of text. Default: `None`
  - **`textsize`**: Size (height) of the text. Default: `120`
  - **`textback`**: Coronal (depth) position of the back of the text. Default: `300`
  - **`textfront`**: Coronal (depth) position of the front of the text. Default: `330`
  - **`textlinelength`**: Length of line below the text. Default: `10`

</details>


## Command-Line Interface 🖥️
Process your first brain scan from the terminal by just running
```bash
brain-stl
```
Without providing any arguments, it downloads a publicly available scan and creates a `brain.stl` of it 🧠

Run `brain-stl --help` (+copy&paste into ChatGPT) to understand the usage of custom settings 💡

## Share Your Brains 🧠🌐
Use `create_gif.py` to get a GIF of your model and share it on social media with the hashtag `#brain-stl` 🤗
