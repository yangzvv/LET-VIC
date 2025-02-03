# LET-VIC: LiDAR-based End-to-End Tracking for Vehicle-Infrastructure Cooperation

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch 1.9+](https://img.shields.io/badge/PyTorch-1.9.1%2BCUDA111-red.svg)](https://pytorch.org/)
[![MMCV 1.4.0](https://img.shields.io/badge/MMCV_Full-1.4.0-green.svg)](https://mmcv.readthedocs.io/)

**LET-VIC** is the first LiDAR-based end-to-end tracking framework specifically designed for vehicle-infrastructure cooperative systems. This implementation provides:

- 🚀 Multi-view fusion with temporal alignment
- 📦 Compatibility with V2X-Seq-SPD dataset
- 🛠️ Extensible architecture for cooperative perception

---

## 📥 Installation

### 1. Environment Setup
```bash
# Create and activate conda environment
conda create -n let-vic python=3.8 -y
conda activate let-vic
```

### 2. PyTorch Installation
```bash
# Install CUDA toolkit
conda install cudatoolkit=11.1.1 -c conda-forge

# Install PyTorch with CUDA 11.1
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 \
    -f https://download.pytorch.org/whl/torch_stable.html
```

### 3. Compiler Configuration
```bash
# Verify GCC version (>=5.0 required)
gcc --version

# If system GCC <5.0:
conda install -c omgarcia gcc-6 -y
export PATH=${CONDA_PREFIX}/bin:$PATH
```

### 4. CUDA Path Setup
```bash
# Set CUDA_HOME according to your installation
export CUDA_HOME=/usr/local/cuda-11.1/
```

### 5. MMCV Ecosystem Installation
```bash
# Install mmcv-full
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# Install detection & segmentation packages
pip install mmdet==2.14.0 mmsegmentation==0.14.1

# Install mmdet3d from source
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d && git checkout v0.17.1
pip install -v -e .  # -v for verbose, -e for editable mode
```

### 6. Project Dependencies
```bash
# Install core requirements
cd LET-VIC
pip install -r requirements.txt

# Install Argoverse API
git clone https://github.com/argoverse/argoverse-api.git
cd argoverse-api && pip install -e .
```

---

## 📁 Dataset Preparation

- readme: [data preparation](./tools/spd_data_converter/README.md)

### V2X-Seq-SPD Configuration

#### Directory Structure
```
datasets/
└── V2X-Seq-SPD-Example/
    ├── cooperative-vehicle-infrastructure/
    │   ├── vehicle-side/
    │   └── infrastructure-side/
    └── maps/
```

#### Data Conversion

##### Method 1: Quick conversion (example sequences)
```bash
bash tools/spd_example_converter.sh V2X-Seq-SPD-Example
```

##### Method 2: Custom conversion
```bash
python tools/spd_data_converter/gen_example_data.py \
    --input <path/to/raw_dataset> \
    --output ./datasets/V2X-Seq-SPD-Example \
    --sequences 0010 0016 0018 0022 0023 0025 0029 0030 0032 0033 0034 0035
```

```bash
# Convert to UniAD format
python tools/spd_data_converter/spd_to_uniad.py \
    --data-root ./datasets/V2X-Seq-SPD-Example \
    --save-root ./data/infos/V2X-Seq-SPD-Example \
    --v2x-side vehicle-side

# Convert to nuScenes format
python tools/spd_data_converter/spd_to_nuscenes.py \
    --data-root ./datasets/V2X-Seq-SPD-Example \
    --save-root ./datasets/V2X-Seq-SPD-Example \
    --v2x-side vehicle-side

# Generate map annotations
python tools/spd_data_converter/map_spd_to_nuscenes.py \
    --save-root ./datasets/V2X-Seq-SPD-Example \
    --v2x-side vehicle-side
```

---

## 🚀 Inference & Training

### Quick Inference

```bash
CUDA_VISIBLE_DEVICES=0 ./tools/xet_vic_dist_eval.sh \
    ./projects/configs/let_vic.py \
    ./ckpts/let_vic.pth \
    1  # Number of GPUs
```

### Full Training Pipeline

```bash
# Single-GPU training
python tools/train.py ./projects/configs/let_vic.py \
    --work-dir ./work_dirs/let_vic \
    --cfg-options model.pretrained=./ckpts/init_weights.pth

# Multi-GPU distributed training
./tools/xet_vic_dist_train.sh \
    ./projects/configs/let_vic.py \
    4  # Number of GPUs
```

---

## 📜 License
This project is released under the [Apache License 2.0](https://github.com/OpenDriveLab/UniAD/blob/main/LICENSE). For commercial use, please contact the authors.


