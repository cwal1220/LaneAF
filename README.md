[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/laneaf-robust-multi-lane-detection-with/lane-detection-on-culane)](https://paperswithcode.com/sota/lane-detection-on-culane?p=laneaf-robust-multi-lane-detection-with)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/laneaf-robust-multi-lane-detection-with/lane-detection-on-llamas)](https://paperswithcode.com/sota/lane-detection-on-llamas?p=laneaf-robust-multi-lane-detection-with)

# LaneAF: Robust Multi-Lane Detection with Affinity Fields

[Paper link](http://cvrr.ucsd.edu/publications/2021/LaneAF.pdf)

[Video results](https://youtube.com/playlist?list=PLUebh5NWCQUZv8IXYOVNM5SuRYQzScW5P)

[Target definition](./TARGET_DEFINITION.md)

![VAF](assets/result_zoomed_VAF.png)
![HAF](assets/result_zoomed_HAF.png)

## Overview
1) [Installation](#installation)
2) [Quick Start](#quick-start)
3) [TuSimple](#tusimple)
4) [CULane](#culane)
5) [Unsupervised Llamas](#unsupervised-llamas)
6) [Pre-trained Weights](#pre-trained-weights)
7) [Citation](#citation)

## Installation
1) Clone this repository
2) Install `uv`
3) Create a modern virtual environment and install dependencies:
```shell
cd LaneAF
uv python install 3.12
uv sync
```

### macOS setup
- Apple Silicon Macs can run `erfnet`, `enet`, and `dla34` with `--device mps`.
- Intel Macs can use the same environment, usually with `--device cpu`.
- The `pyproject.toml` CUDA wheel override only applies on Linux `x86_64`, so macOS installs the regular PyTorch wheels automatically.
- Verify the runtime with:
```shell
uv run python -c "import torch; print(torch.__version__); print('mps', torch.backends.mps.is_available()); print('cuda', torch.cuda.is_available())"
```
- Example training command on Apple Silicon:
```shell
uv run python train_culane.py --dataset-dir=/path/to/CULane/ --backbone=dla34 --device=mps
```

### Linux x86_64 / CUDA setup
- Linux `x86_64` uses the official PyTorch `cu121` wheels.
- This repository is pinned to Python `3.12` because the `3.14` wheels currently resolve to a newer CUDA runtime that may be incompatible with older NVIDIA drivers.
- On NVIDIA systems with drivers roughly matching CUDA `12.2`, `uv sync` should install a working CUDA-enabled PyTorch build.
- You can verify GPU availability with:
```shell
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
```
- Example training command on Linux with NVIDIA GPU:
```shell
uv run python train_culane.py --dataset-dir=/path/to/CULane/ --backbone=dla34 --device=cuda
```

### Device notes
- The repository now supports `--device auto|cpu|cuda|mps`.
- `--device auto` prefers `cuda`, then `mps`, then `cpu`.
- On Apple Silicon / macOS, use `--device mps`.
- `dla34`, `erfnet`, and `enet` all run without any extra native extensions.
- `dla34` still loads the standard ImageNet-pretrained DLA-34 backbone weights, but its LaneAF upsampling path now uses standard `Conv2d` blocks instead of `DCNv2`.

### Checkpoints and Export
- Training snapshots are saved as `.pth` files.
- `export_onnx.py` exports a fixed-batch ONNX graph with batch size `1`, which is usually the right shape contract for NPU deployment.
- Example:
```shell
uv run python export_onnx.py \
  --snapshot /path/to/model.pth \
  --output /path/to/model_b1.onnx \
  --backbone dla34 \
  --height 288 \
  --width 832 \
  --clean-identity
```

## Quick Start
1) Install the environment:
```shell
cd LaneAF
uv python install 3.12
uv sync
```

2) Download the CULane archive.
- One working source is Kaggle: https://www.kaggle.com/datasets/manideep1108/culane
- This guide assumes the archive is available at `/home/chan/Downloads/CULane.zip`.

3) Extract the dataset:
```shell
mkdir -p data/CULane
unzip /home/chan/Downloads/CULane.zip -d data/CULane
```

4) Start training.
- Apple Silicon / macOS:
```shell
uv run python train_culane.py --dataset-dir=data/CULane --backbone=dla34 --device=mps
```
- Linux `x86_64` with NVIDIA GPU:
```shell
uv run python train_culane.py --dataset-dir=data/CULane --backbone=dla34 --device=cuda
```
- On this machine, the full CULane training command is:
```shell
uv run python train_culane.py --dataset-dir=/home/chan/LaneAF/data/CULane_full --backbone=dla34 --device=cuda --random-transforms
```

5) Monitor training with TensorBoard:
```shell
uv run python -m tensorboard.main --logdir experiments/culane --host 0.0.0.0 --port 6006
```

6) Export a trained checkpoint to ONNX for batch-1 deployment:
```shell
uv run python export_onnx.py \
  --snapshot /path/to/model.pth \
  --output /path/to/model_b1.onnx \
  --backbone dla34 \
  --height 288 \
  --width 832 \
  --clean-identity
```

## TuSimple
The entire [TuSimple dataset](https://github.com/TuSimple/tusimple-benchmark/issues/3) should be downloaded and organized as follows:
```plain
└── TuSimple/
    ├── clips/
    |   └── .
    |   └── .
    ├── label_data_0313.json
    ├── label_data_0531.json
    ├── label_data_0601.json
    ├── test_tasks_0627.json
    ├── test_baseline.json
    └── test_label.json
```
The model requires ground truth segmentation labels during training. You can generate these for the entire dataset as follows:
```shell
uv run python datasets/tusimple.py --dataset-dir=/path/to/TuSimple/
```

### Training
LaneAF models can be trained on the TuSimple dataset as follows:
```shell
uv run python train_tusimple.py --dataset-dir=/path/to/TuSimple/ --backbone=erfnet --device=mps --random-transforms
```
Other supported backbones are `enet` and `dla34`.

Config files, logs, results and snapshots from running the above scripts will be stored in the `LaneAF/experiments/tusimple` folder by default.

### Inference
Trained LaneAF models can be run on the TuSimple test set as follows:
```shell
uv run python infer_tusimple.py --dataset-dir=/path/to/TuSimple/ --snapshot=/path/to/trained/model/snapshot --device=mps --save-viz
```
This will generate outputs in the TuSimple format and also produce benchmark metrics using their [official implementation](https://github.com/TuSimple/tusimple-benchmark/tree/master/doc/lane_detection).

### Results
| Backbone | F1-score | Accuracy |   FP   |   FN   |
|:--------:|:--------:|:--------:|:------:|:------:|
|  DLA-34  |  96.4891 |  95.6172 | 0.0280 | 0.0418 |
|  ERFNet  |  94.9465 |  95.2978 | 0.0550 | 0.0465 |
|   ENet   |  92.8905 |  94.7271 | 0.0885 | 0.0560 |

## CULane
The entire [CULane dataset](https://xingangpan.github.io/projects/CULane.html) should be downloaded and organized as follows:
```plain
└── CULane/
    ├── driver_*_*frame/
    ├── laneseg_label_w16/
    ├── laneseg_label_w16_test/
    └── list/
```

### Training
LaneAF models can be trained on the CULane dataset as follows:
```shell
uv run python train_culane.py --dataset-dir=/path/to/CULane/ --backbone=erfnet --device=mps --random-transforms
```
Other supported backbones are `enet` and `dla34`.
Examples:
```shell
uv run python train_culane.py --dataset-dir=/path/to/CULane/ --backbone=dla34 --device=mps --random-transforms
uv run python train_culane.py --dataset-dir=/path/to/CULane/ --backbone=dla34 --device=cuda --random-transforms
```

Config files, logs, results and snapshots from running the above scripts will be stored in the `LaneAF/experiments/culane` folder by default.
`train_culane.py` also writes TensorBoard event files under `output_dir/tensorboard`.

### Inference
Trained LaneAF models can be run on the CULane test set as follows:
```shell
uv run python infer_culane.py --dataset-dir=/path/to/CULane/ --snapshot=/path/to/trained/model/snapshot --device=mps --save-viz
```
This will generate outputs in the CULane format. You can then use their [official code](https://github.com/XingangPan/SCNN) to evaluate the model on the CULane benchmark.

### Results
| Backbone | Total | Normal | Crowded | Dazzle | Shadow | No line | Arrow | Curve | Cross | Night |
|:--------:|:-----:|:------:|:-------:|:------:|:------:|:-------:|:-----:|:-----:|:-----:|:-----:|
|  DLA-34  | 77.41 |  91.80 |  75.61  |  71.78 |  79.12 |  51.38  | 86.88 | 71.70 |  1360 | 73.03 |
|  ERFNet  | 75.63 |  91.10 |  73.32  |  69.71 |  75.81 |  50.62  | 86.86 | 65.02 |  1844 | 70.90 |
|   ENet   | 74.24 |  90.12 |  72.19  |  68.70 |  76.34 |  49.13  | 85.13 | 64.40 |  1934 | 68.67 |

## Unsupervised Llamas
The [Unsupervised Llamas dataset](https://unsupervised-llamas.com/llamas/index) should be downloaded and organized as follows:
```plain
└── Llamas/
    ├── color_images/
    |   ├── train/
    |   ├── valid/
    |   └── test/
    └── labels/
        ├── train/
        └── valid/
```

### Training
LaneAF models can be trained on the Llamas dataset as follows:
```shell
uv run python train_llamas.py --dataset-dir=/path/to/Llamas/ --backbone=erfnet --device=mps --random-transforms
```
Other supported backbones are `enet` and `dla34`.

Config files, logs, results and snapshots from running the above scripts will be stored in the `LaneAF/experiments/llamas` folder by default.

### Inference
Trained LaneAF models can be run on the Llamas test set as follows:
```shell
uv run python infer_llamas.py --dataset-dir=/path/to/Llamas/ --snapshot=/path/to/trained/model/snapshot --device=mps --save-viz
```
This will generate outputs in the CULane format and Llamas format for the Lane Approximations benchmark. 
Note that the results produced in the Llamas format could be inaccurate because we *guess* the IDs of the indivudal lanes. 

### Results
| Backbone | F1-score | Precision | Recall |   TP  |  FP  |  FN  |
|:--------:|:--------:|:---------:|:------:|:-----:|:----:|:----:|
|  DLA-34  |   96.01  |   96.91   |  95.26 | 71793 | 2291 | 3576 |
|  ERFNet  |     NA   |     NA    |    NA  |   NA  |  NA  |  NA  |
|   ENet   |     NA   |     NA    |    NA  |   NA  |  NA  |  NA  |

## Pre-trained Weights 
You can download our pre-trained model weights using [this link](https://drive.google.com/file/d/1GJoVQfDyxhUT8Y5EqTRV9PX3WWckfxWG/view?usp=sharing).

## Citation
If you find our code and/or models useful in your research, please consider citing the following papers:

    @article{abualsaud2021laneaf,
    title={LaneAF: Robust Multi-Lane Detection with Affinity Fields},
    author={Abualsaud, Hala and Liu, Sean and Lu, David and Situ, Kenny and Rangesh, Akshay and Trivedi, Mohan M},
    journal={arXiv preprint arXiv:2103.12040},
    year={2021}
    }
