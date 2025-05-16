
# 🧱 MMRotate Brick Kiln Detection

This repository contains code and configurations for detecting brick kilns using rotated object detection models based on the [MMRotate](https://github.com/open-mmlab/mmrotate) framework. Brick kiln identification is critical for environmental monitoring and policy enforcement. We evaluate multiple obb object detectors on a custom satellite imagery dataset curated for this task, and provide training, inference, and analysis tools.


## Results and Models
    
| Category      | Method           | Publication | Backbone  | BBox | CA mAP50 | CFCBK  | FCBK   | Zigzag | Configs | Models | Logs |
|:-------------:|:-----------------|:-----------:|:---------:|:----:|:--------:|:------:|:------:|:------:|:-------:|:------:|:----:|
| Two-stage     | PSC              | CVPR-23     | Res50     | OBB  | 27.41    | 0.38   | 13.83  | 17.03  | [cfg]() | [model]() | [log]() |
|               | H2RBox           | ICLR-23     | Res50     | OBB  | 47.01    | 24.93  | 30.27  | 31.02  | [cfg]() | [model]() | [log]() |
|               | RoI Transformer  | CVPR-19     | Swin-T    | OBB  | 61.65    | 45.31  | 43.75  | 47.46  | [cfg]() | [model]() | [log]() |
| One-stage     | Rotated FCOS     | ICCV-19     | Res50     | OBB  | 15.62    | 12.72  | 8.48   | 9.99   | [cfg]() | [model]() | [log]() |
|               | DCFL             | CVPR-23     | Res50     | OBB  | 16.01    | 9.07   | 6.51   | 12.97  | [cfg]() | [model]() | [log]() |
|               | CSL              | ECCV-20     | Res50     | OBB  | 16.48    | 0.18   | 9.31   | 8.37   | [cfg]() | [model]() | [log]() |
|               | Rotated RetinaNet| ICCV-17     | Res50     | OBB  | 34.37    | 2.81   | 17.27  | 22.55  | [cfg]() | [model]() | [log]() |
|               | Rotated ATSS     | CVPR-20     | Res50     | OBB  | 38.79    | 18.68  | 20.49  | 25.27  | [cfg]() | [model]() | [log]() |
|               | GWD              | ICML-21     | Res50     | OBB  | 41.70    | 0.17   | 22.21  | 25.12  | [cfg]() | [model]() | [log]() |
|               | R³Det            | AAAI-21     | Res50     | OBB  | 43.70    | 0.17   | 24.89  | 28.54  | [cfg]() | [model]() | [log]() |
|               | S²A-Net          | TGRS-21     | Res50     | OBB  | 54.28    | 32.10  | 32.21  | 39.85  | [cfg]() | [model]() | [log]() |
|               | ConvNeXt         | CVPR-22     | Res50     | OBB  | 66.19    | 34.33  | 41.84  | 43.53  | [cfg]() | [model]() | [log]() |
|               | YOLOv11L-OBB     | arXiv-24    | CSPDr53   | OBB  | 75.20    | _58.57_ | 55.43  | 53.90  | [cfg]() | [model]() | [log]() |
|               | YOLOv12L         | arXiv-25    | CSPDr53   | AA   | 79.54    | 58.00  | 60.19  | 54.83  | [cfg]() | [model]() | [log]() |
|               | YOLOv8L-WORLDv2  | CVPR-24     | CSPDr53   | AA   | _80.74_  | 57.78  | _61.28_ | _56.89_ | [cfg]() | [model]() | [log]() |
| DETR-Based    | DETA             | ICCV-23     | Res50     | AA   | 65.34    | 44.01  | 47.56  | 55.21  | [cfg]() | [model]() | [log]() |
|               | RFDETR           | arXiv-25    | Dinov2    | AA   | 79.64    | 64.26  | 63.68  | 64.25  | [cfg]() | [model]() | [log]() |
|               | RTDETR           | CVPR-24     | Res101    | AA   | **87.53** | **63.03** | **68.60** | **64.30** | [cfg]() | [model]() | [log]() |


## ⚙️ Installation

[MMRotate](https://github.com/open-mmlab/mmrotate) depends on [PyTorch](https://pytorch.org), [MMCV](https://github.com/open-mmlab/mmcv), and [MMDetection](https://github.com/open-mmlab/mmdetection). Please refer to the [Install Guide](https://mmrotate.readthedocs.io/en/latest/install.html) for more detailed instructions. Below are quick steps for installation.


1. Set up the environment and install MMRotate.
    ```
    conda create -n open-mmlab python=3.8 -y
    conda activate open-mmlab

    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
    # conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
    pip uninstall pycocotools

    git clone https://github.com/open-mmlab/mmrotate.git
    cd mmrotate
    pip install -r requirements/build.txt
    pip install -v -e . --user

    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple mmengine
    mim install mmengine
    ```

2. Clone this repo and copy configs and files to mmrotate folder.
    ```
    cd ../
    git clone https://github.com/rishabh-mondal/NeurIPS_2025.git
    cp -r NeurIPS_2025/mmrotate/mmrotate_brickkiln/* mmrotate/
    cd mmrotate
    ```




## 🚀 Example Usage

### 🏋️ Training
[dist_train.sh](./mmrotate_brickkiln/tools/dist_train.sh) is a script for distributed training. It can be used to train models on multiple GPUs. The script takes two arguments: the configuration file and which GPUs to use separated by commas.
```sh
# bash tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUMs}
bash tools/dist_train.sh 'configs/roi_trans/roi_trans_swin_tiny_fpn_1x_brickkiln_le90.py' 2
```


### 🔎 Inference

#### Single Image Inference
To run inference on a single image, edit the variables at the top of [image_inference.py](./mmrotate_brickkiln/image_inference.py) to set the config file, checkpoint path, and image path. Example:

```python
# In image_inference.py, set these variables:
config_file = 'configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_brickkiln_le90.py'
checkpoint_file = 'work_dirs/oriented_rcnn_r50_fpn_1x_brickkiln_le90_first/latest.pth'
image_path = '../data/sentinel/test/images/28.2090_77.4057.png'  # Replace with your image
```
Then simply run:
```sh
python image_inference.py
```

#### Batch and Multi-Epoch Inference
To run inference on a folder of images or across multiple epochs, manually edit the variables at the bottom of [folder_inference.py](./mmrotate_brickkiln/image_inference.py) or [folder_inference_multi_epoch.py](./mmrotate_brickkiln/folder_inference_multi_epoch.py) to set the config file, checkpoint directory, image directory and selected epochs. Example:

```python
# In folder_inference.py, uncomment and set these variables:
config_file = 'configs/roi_trans/roi_trans_swin_tiny_fpn_1x_brickkiln_le90.py'
checkpoint_folder = 'work_dirs/roi_trans_swin_tiny_fpn_1x_brickkiln_le90'
image_dir = '../data/stratified_split/test/images'  # Directory containing images
modelname = 'test_roi_trans_swin_tiny_fpn_1x_brickkiln_le90'
```
Then simply run:
```sh
python folder_inference.py                  # for single epoch inference
# or
python folder_inference_multi_epoch.py      # for multi-epoch inference
```

### 📈 Log Analysis
Refer to the [MMRotate Documentation](https://mmrotate.readthedocs.io/en/latest/useful_tools.html) for more details on log analysis. [plot_loss](./mmrotate_brickkiln/loss_plots/plot_loss.md) contains few commands to plot loss curves for different models.



## 🗂️ Folder Structure and Utilities

The `mmrotate_brickkiln` folder contains several scripts, notebooks, and resources to support training, evaluation, and analysis for brick kiln detection:

### 🧰 Configs and Custom Dataset

- `configs/`: Contains config files for various MMRotate models (Oriented R-CNN, ROI Transformer, ReDet, S2ANet, etc.) tailored for brick kiln detection.
- `mmrotate/datasets/brickkiln.py`: Custom dataset definition for brick kiln detection, enabling integration with MMRotate's training and evaluation pipelines.



### 📊 Evaluation and Analysis

- `maps/`: Contains mAP (mean Average Precision) plots and CSVs for different models and checkpoints.
- `loss_plots/`: Contains loss curve plots for different models for loss analysis.
- [map_dota_all_epoch.ipynb](./mmrotate_brickkiln/map_dota_all_epoch.ipynb): Jupyter notebook for calculating mAP across all epochs for DOTA-format results.
- [map_dota_single_epoch.ipynb](./mmrotate_brickkiln/map_dota_single_epoch.ipynb): Jupyter notebook for calculating mAP for a single epoch for DOTA-format results.
- [compare_img.ipynb](./mmrotate_brickkiln/compare_img.ipynb): Jupyter notebook for visually comparing images from two directories (e.g., ground truth vs. predictions).



## 🧹 Remove the environment (Optional cleanup)
```sh
conda deactivate
conda env remove -n open-mmlab
```