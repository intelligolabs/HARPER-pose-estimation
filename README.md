# Pose estimation baseline for the 3D Human Pose Estimation benchmark of the HARPER dataset (IROS 2024) 

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a> <a href="https://bostondynamics.com/products/spot/"><img src="https://img.shields.io/badge/Boston%20Dynamics%20-Spot-yellow"></a> [![arXiv](https://img.shields.io/badge/arXiv-2403.14447-b31b1b.svg)](https://arxiv.org/abs/2403.14447) <a href="https://intelligolabs.github.io/HARPER/"><img alt="Project" src="https://img.shields.io/badge/-Project%20Page-lightgrey?logo=Google%20Chrome&color=informational&logoColor=white"></a>

This repository contains the code for the baseline of the 3D Human Pose Estimation benchmark on the [HARPER](https://intelligolabs.github.io/HARPER/) dataset.
The baseline is based on the [HRNet](https://arxiv.org/abs/1908.07919) architecture and uses the depth maps captured by the [Spot](https://bostondynamics.com/products/spot/) to estimate the 3D pose of the partially-visible human body.

## Quick start
The quick start guide will be available soon.

### Installation
You can install the required packages following the steps [here](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch).  
You can find the pretrained HRNet model [here](https://univr-my.sharepoint.com/:u:/g/personal/andrea_toaiari_univr_it/EXVXjMoApr5FvECihR220xwB8vg0dC5OMqOCeZWtypaV1g?e=oJ1kXE). To use it, modify the `TEST.MODEL_FILE` parameter in the config file (`experiments/harper/hrnet/w32_256x256_adam_lr1e-3_harper.yaml`) with the correct path.

### Data preparation
Follow the steps in the HARPER [official repository](https://github.com/intelligolabs/HARPER) to download the dataset and prepare the data.   
Modify the `DATASET.ROOT` parameter in the config file with the correct path.

## Credits
This code is based on the [HRNet](https://arxiv.org/abs/1908.07919) architecture, forking this [implementation](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch).

## Citation
If you use this code in your research, please cite the following paper *(IROS 2024 citation coming soon)*:
```
@article{avogaro2024exploring,
    title={Exploring 3D Human Pose Estimation and Forecasting from the Robot's Perspective: The HARPER Dataset},
    author={Avogaro, Andrea and Toaiari, Andrea and Cunico, Federico and Xu, Xiangmin and Dafas, Haralambos and Vinciarelli, Alessandro and Li, Emma and Cristani, Marco},
    journal={arXiv e-prints},
    pages={arXiv--2403},
    year={2024}
}
```
