# LET-VIC
LET-VIC is the first LiDAR-based End-to-end Tracking framework for Vehicle-Infrastructure Cooperation.

---
## Installation

a. Env: Create a conda virtual environment and activate it.

```bash
conda create -n xet-vic python=3.8 -y
conda activate xet-vic
```

b. Torch: Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).

```bash
conda install cudatoolkit=11.1.1 -c conda-forge
# We use cuda-11.1 by default
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# Recommended torch>=1.9
```

c. GCC: Make sure gcc>=5 in conda env.

```bash
# If gcc is not installed:
# conda install -c omgarcia gcc-6 # gcc-6.2

export PATH=YOUR_GCC_PATH/bin:$PATH
# Eg: export PATH=/mnt/gcc-5.4/bin:$PATH
```

d. CUDA: Before installing MMCV family, you need to set up the CUDA_HOME (for compiling some operators on the gpu).

```bash
export CUDA_HOME=YOUR_CUDA_PATH/
# Eg: export CUDA_HOME=/mnt/cuda-11.1/
```

e. Install mmcv-full.

```bash
pip install mmcv-full==1.4.0
# If it's not working, try:
# pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

f. Install mmdet and mmseg.

```bash
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

g. Install mmdet3d from source code.

```bash
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1
pip install scipy==1.7.3
pip install scikit-image==0.20.0
pip install -v -e .
```

h. Install XET-VIC.

```bash
cd XET-VIC
pip install -r requirements.txt
```

i. Install Argoverse API

```bash
git clone https://github.com/argoverse/argoverse-api.git
cd argoverse-api
pip install -e .
```
---

## Dataset Preparation
- readme: [data preparation](./tools/spd_data_converter/README.md)

<<<<<<< HEAD
- run

```bash
python tools/spd/gen_example_data.py \
   --input /path/to/V2X-Seq-SPD \
   --output ./datasets/V2X-Seq-SPD-Example \
   --sequences 0036 0037 0040 0041 0047 0058 0059
=======
- run: method I

```bash
bash tools/spd_example_converter.sh V2X-Seq-SPD-Example
```

- run: method II

```bash
    python tools/spd_data_converter/gen_example_data.py
        --input /data/ad_sharing/datasets/v2x-seq/SPD/cooperative-vehicle-infrastructure
        --output ./datasets/V2X-Seq-SPD-Example
        --sequences 0010 0016 0018 0022 0023 0025 0029 0030 0032 0033 0034 0035 0014 0015 0017 0020 0021
```

```bash
    python tools/spd_data_converter/spd_to_uniad.py
        --data-root ./datasets/V2X-Seq-SPD-Example
        --save-root ./data/infos/V2X-Seq-SPD-Example
        --v2x-side 'vehicle-side`
```
```bash
    python tools/spd_data_converter/spd_to_nuscenes.py
        --data-root ./datasets/V2X-Seq-SPD-Example
        --save-root ./datasets/V2X-Seq-SPD-Example
        --v2x-side vehicle-side
```
```bash
    python tools/spd_data_converter/map_spd_to_nuscenes.py
        --save-root ./datasets/V2X-Seq-SPD-Example
        --v2x-side vehicle-side
```
## Inference
```bash
    CUDA_VISIBLE_DEVICES=${GPU_ID} ./tools/xet_vic_dist_eval.sh ./projects/configs/let_vic.py ./ckpts/let_vic.pth 1
```
