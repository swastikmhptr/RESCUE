## RESCUE: Reasoning over Earth Scenes for coordinated UAS-UGS Exploration

![IIR Project Diagram](assets/iir_project_diag.png)

### Submodules

This repo vendors **[lseg-minimal](https://github.com/krrish94/lseg-minimal.git)** (dense CLIP / LSeg features) and **[map-anything](https://github.com/facebookresearch/map-anything.git)** (MapAnything 3D reconstruction) as Git submodules under `lseg-minimal/` and `map-anything/`.

### Install

1. **Pull submodules** (from the RESCUE repo root):
   - New clone: `git clone --recurse-submodules <RESCUE-repo-url>`
   - Existing clone: `git submodule update --init --recursive`

2. **Conda environment**

   ```bash
   conda create -n rescue python=3.12 -y
   conda activate rescue
   ```

3. **Dependencies**

   ```bash
   pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   pip install git+https://github.com/openai/CLIP.git
   ```

4. **LSeg**:

   ```bash
   cd lseg-minimal
   python setup.py build develop
   cd ..
   ```

   LSeg checkpoint ([Google Drive](https://drive.google.com/file/d/1jGDqefX057MFvYIZN25khAhdS2AGFjjA/view?usp=drive_link)):

   ```bash
   pip install gdown
   mkdir -p generated
   gdown 1jGDqefX057MFvYIZN25khAhdS2AGFjjA -O generated/lseg_minimal_e200.ckpt
   ```

5. **MapAnything**

   ```bash
   cd map-anything
   ## use pip install -e ".[all]" for all dependencies to run vggt etc
   pip install -e . 
   cd ..
   ```

6. **Download Renders**

   The dataset renders can be downloaded using

   ```bash
   pip install gdown
   mkdir -p generated/renders
   gdown --folder https://drive.google.com/drive/folders/1zqisevgIJ_X6RczFZE54lGKOAFn1UheX -O generated/renders
   ```

   This will download all files in the `renders` folder to `generated/renders/`.

7. **`rescue` as a library** (from the RESCUE repo root, after the steps above):

   ```bash
   pip install -e .
   ```
