# Recognition Tasks
Various recognition tasks solved in deep learning frameworks.

Tasks may include:
* Image Segmentation
* Object detection
* Graph node classification
* Image super resolution
* Disease classification
* Generative modelling with StyleGAN and Stable Diffusion





## Environment setup
1. Install Python 3.11 (recommended).
2. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate

### Installing from requirements text file
pip install -r requirements.txt


### Instealling manually

pip install torch==2.4.0 torchvision==0.19.0 numpy==1.26.4 matplotlib==3.9.0 nibabel==5.3.1 scipy==1.13.1 pillow==10.4.0 tqdm==4.66.5 ipywidgets==8.1.2 notebook==7.2.1 jupyterlab==4.2.5

### Minimal install

pip install torch==2.4.0 torchvision==0.19.0 numpy==1.26.4 nibabel==5.3.1 scipy==1.13.1 pillow==10.4.0 tqdm==4.66.5 matplotlib==3.9.0

### Full Install

pip install torch==2.4.0 torchvision==0.19.0 numpy==1.26.4 nibabel==5.3.1 scipy==1.13.1 pillow==10.4.0 tqdm==4.66.5 matplotlib==3.9.0 ipywidgets==8.1.2 notebook==7.2.1 jupyterlab==4.2.5



## Quick setup (no requirements.txt)
```bash
pip install torch==2.4.0 torchvision==0.19.0 numpy==1.26.4 matplotlib==3.9.0 nibabel==5.3.1 scipy==1.13.1 pillow==10.4.0 tqdm==4.66.5 ipywidgets==8.1.2 notebook==7.2.1 jupyterlab==4.2.5


## Loading whole environment with yaml file

## Option A gpu
# create env (one time)
conda env create -f environment-gpu.yml
conda activate comp3710-gpu

# verify GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# run
python recognition/Project2/train.py   --data_root "C:\...\keras_slices_data" --epochs 20 --batch_size 4
python recognition/Project2/predict.py --data_root "C:\...\keras_slices_data" --save_pngs --show_samples




##Option b cpu only

conda env create -f environment-cpu.yml
conda activate comp3710-cpu

# run (slower but works everywhere)
python recognition/Project2/predict.py --data_root "C:\...\keras_slices_data" --max_test_slices 50 --save_pngs


PatternAnalysis-2025/
├── recognition/
│   ├── Project2/
│   │   ├── train.py
│   │   ├── predict.py
│   │   ├── dataset.py
│   │   ├── modules.py
│   │   └── utils.py
│   └── ...
├── environment-gpu.yml
├── environment-cpu.yml
└── (your README, .gitignore, etc.)
