# SentinelKilnDB  
### A Large-Scale Dataset and Benchmark for OBB Brick Kiln Detection in South Asia Using Satellite Imagery  
**[NeurIPS 2025 Submission]**

---

## Dataset Overview — *SentinelKilnDB*

| Attribute        | Details                                 |
|------------------|------------------------------------------|
| **Dataset**     | SentinelKilnDB                          |
| **Size**        | ~4.00 GB                                |
| **Images**      | 78,707 RGB (10m resolution)             |
| **Satellite**    | Sentinel-2                              |
| **Time Period**  | Sept 2023 – Feb 2024                    |
| **Annotations**  | 105,933 Oriented Bounding Boxes (OBBs) |
| **Classes**      | 3 — CFCBK, FCBK, Zigzag                 |
| **Format**       | DOTA, YOLO-OBB, YOLO-AA                |
| **License**      | CC BY-NC-SA 4.0                         |

---
---

## Sample Images

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/rishabh-mondal/NeurIPS_2025/blob/main/figures/sent_1.png" alt="Image 1" width="300"/><br/>
      <sub><b>Image 1</b></sub>
    </td>
    <td align="center">
      <img src="https://github.com/rishabh-mondal/NeurIPS_2025/blob/main/figures/sent_2.png" alt="Image 2" width="300"/><br/>
      <sub><b>Image 2</b></sub>
    </td>
    <td align="center">
      <img src="https://github.com/rishabh-mondal/NeurIPS_2025/blob/main/figures/sent_3.png" alt="Image 3" width="300"/><br/>
      <sub><b>Image 3</b></sub>
    </td>
  </tr>
</table>

---

## Dataset Split

Stratified by class to ensure balanced representation across training, validation, and test sets.

| Split       | Images  | Annotations | CFCBK | FCBK  | Zigzag |
|-------------|---------|-------------|-------|-------|--------|
| Train       | 47,214  | 47,214      | 2,032 | 34,292| 27,463 |
| Validation  | 15,738  | 15,738      |   649 | 11,339|  9,054 |
| Test        | 15,738  | 15,738      |   647 |  9,141| 11,312 |
| **Total**   | 78,707  | 105,933     | 3,328 | 54,772| 47,829 |

---


## Data Downloading Process and Preprocessing Notebooks

This section summarizes the key scripts used for downloading Sentinel-2 tiles, handling missing data, preprocessing labels, and splitting datasets. All scripts are located in the [`data_scripts`](https://github.com/rishabh-mondal/NeurIPS_2025/tree/main/data_scripts) directory.

---

### Downloading & Tile Management

| Script | Description |
|--------|-------------|
| [`sentinel_tile_bulk_download.py`](https://github.com/rishabh-mondal/NeurIPS_2025/blob/main/data_scripts/sentinel_tile_bulk_download.py) | Downloads Sentinel-2 tiles in bulk based on region and time range. |
| [`finding_missing_tiles.py`](https://github.com/rishabh-mondal/NeurIPS_2025/blob/main/data_scripts/finding_missing_tiles.py) | Detects tiles that are missing or not downloaded properly. |
| [`run_missing_tiles_download.py`](https://github.com/rishabh-mondal/NeurIPS_2025/blob/main/data_scripts/run_missing_tiles_download.py) | Re-downloads missing or corrupted tiles. |

---

### Preprocessing & Visualization

| Script | Description |
|--------|-------------|
| [`tile_processing.py`](https://github.com/rishabh-mondal/NeurIPS_2025/blob/main/data_scripts/tile_processing.py) | Splits raw Sentinel-2 tiles into fixed-size patches, preprocesses labels, and visualizes patches. |
| [`label_format_coversion.py`](https://github.com/rishabh-mondal/NeurIPS_2025/blob/main/data_scripts/label_format_coversion.py) | Converts labels into YOLO, DOTA, and YOLO-AA formats. |

---

### Dataset Splitting

| Script | Description |
|--------|-------------|
| [`data_splits.py`](https://github.com/rishabh-mondal/NeurIPS_2025/blob/main/data_scripts/data_splits.py) | Performs stratified train/val/test split based on label distribution. |

---

> **Tip**: Each script is modular and can be executed independently, making the pipeline flexible and customizable.

---

# 🧩 Unified Setup Guide — One-Cell Environment

> A **single Conda environment** for multiple cutting-edge object detection frameworks, enabling reproducible and seamless development.

---

## Frameworks Included

| Framework   | Description                                     | Source Link |
|-------------|-------------------------------------------------|-------------|
| **MMRotate** | Rotated bounding box detection                  | [MMRotate](https://github.com/open-mmlab/mmrotate) |
| **Ultralytics** | YOLOv8 / YOLOv11-OBB / RT-DETR / YOLOv12        | [Ultralytics](https://github.com/ultralytics/ultralytics) |
| **RFDETR**  | Custom Transformer-based detector               | [RFDETR](https://github.com/rishabh-mondal/NeurIPS_2025) |
| **DETA**    | Hugging Face's DETR variant                     | [DETA](https://huggingface.co/docs/transformers/main/en/model_doc/deta) |

---

## Environment Setup: `oncell`

**Requirements:**
- Python ≥ 3.8
- CUDA 11.1-enabled GPU

All frameworks are installed in a single Conda environment named `oncell`.

---

## Installation Steps

<!-- <details>
<summary><strong>Expand for step-by-step instructions</strong></summary> -->

```bash
# 1. Create and activate a Conda environment
conda create -n oncell python=3.8 -y
conda activate oncell

# 2. Install PyTorch with CUDA 11.1
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# 3. Install MMCV (for MMRotate)
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# 4. Remove conflicting packages
pip uninstall -y pycocotools

# 5. Install MMRotate
git clone https://github.com/open-mmlab/mmrotate.git
cd mmrotate
pip install -r requirements/build.txt
pip install -v -e . --user
cd ..

# 6. Install MMEngine
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple mmengine
mim install mmengine

# 7. Install Ultralytics
pip install ultralytics

# 8. Install RFDETR
git clone https://github.com/rishabh-mondal/NeurIPS_2025.git
cd NeurIPS_2025/rfdetr
pip install -r requirements.txt
cd ../..

# 9. Install Hugging Face Transformers (for DETA)
pip install transformers

```

---

## Traing the models
mmrotate models can be trained using the following command:

Clone this repo and copy configs and files to mmrotate folder.

cd ../
git clone https://github.com/rishabh-mondal/NeurIPS_2025.git
cp -r NeurIPS_2025/mmrotate/mmrotate_brickkiln/* mmrotate/
cd mmrotate

## Example Usage

# ===== MMRotate Training =====
# Syntax: bash tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUMs}
# Example: Train on 4 GPUs
bash tools/dist_train.sh configs/rotated-retinanet-rbox-le90_r50_fpn_psc-dual_amp-1x_dota.py 0,1,2,3


# ===== Ultralytics Model Training =====
cd training_scripts
chmod +x train.sh
./train.sh


# ===== RFDETR Model Training =====
cd training_scripts
export CUDA_VISIBLE_DEVICES=0
nohup python rfdetr_train.py > ./logs/rfdetr_large/sentinel_data.log 2>&1 &


# ===== DETA Model Training =====
cd training_scripts
export CUDA_VISIBLE_DEVICES=0
nohup python deta_train.py > ./logs/deta_large/sentinel_data.log 2>&1 &





---
## 📊 Results and Benchmarks

> 🟩 **Highest score** per column is highlighted  
> 🔽 Click on a category to expand its models  
> 🏷️ *Badges* indicate model families for quick reference

---

<details>
<summary><strong>🧠 Two-Stage Models</strong></summary>

| 🏷️ Model             | 📄 Paper  | Backbone | BBox | CA mAP50 | CFCBK | FCBK | Zigzag | ⚙️ Config | 💾 Model |
|----------------------|-----------|-------------|---------|-------------|----------|--------|-----------|-----------|----------|
| PSC                  | CVPR-23   | Res50       | OBB     | 27.41       | 0.38     | 13.83  | 17.03     | [📄 Config](https://github.com/rishabh-mondal/NeurIPS_2025/blob/main/configs/psc/rotated-retinanet-rbox-le90_r50_fpn_psc-dual_amp-1x_dota.py) | [🔗 Model](https://drive.google.com/file/d/1yGB0_fcGndLI9uCf678tE3OiHAYi3jds/view?usp=drive_link) |
| H2RBox               | ICLR-23   | Res50       | OBB     | 47.01       | 24.93    | 30.27  | 31.02     | [Config](https://github.com/rishabh-mondal/NeurIPS_2025/blob/main/mmrotate_brickkiln/configs/h2rbox-le90_r50_fpn_adamw-1x_dota.py) | [Model](https://drive.google.com/file/d/18Flaofo6yeTioR_1Xo6T7Hnfh94al9f9/view?usp=drive_link) |
| 🟩 **RoI Transformer** | CVPR-19 | Swin-T      | OBB     | 🟩 **61.65** | 🟩 **45.31** | 🟩 **43.75** | 🟩 **47.46** | [Config](https://github.com/rishabh-mondal/NeurIPS_2025/blob/main/mmrotate_brickkiln/configs/roi_trans_r50_fpn_1x_brickkiln_le90.py) | [Model](https://drive.google.com/file/d/1suX01Id1UR6QD19EZtx8WcM5jKgNLfGq/view?usp=drive_link) |

</details>

---

<details>
<summary><strong>⚡ One-Stage Models</strong></summary>

| 🏷️ Model             | 📄 Paper  | Backbone | BBox | CA mAP50 | CFCBK | FCBK | Zigzag | ⚙️ Config | 💾 Model |
|----------------------|-----------|-------------|---------|-------------|----------|--------|-----------|-----------|----------|
| Rotated FCOS         | ICCV-19   | Res50       | OBB     | 15.62       | 12.72    | 8.48   | 9.99      | [Config](https://github.com/rishabh-mondal/NeurIPS_2025/blob/main/mmrotate_brickkiln/configs/rotated-fcos-le90_r50_fpn_1x_dota.py) | [Model](https://drive.google.com/file/d/14VM3UybLOgL46Q__cqdWcxGV5_HU9nqI/view?usp=drive_link) |
| DCFL                 | CVPR-23   | Res50       | OBB     | 16.01       | 9.07     | 6.51   | 12.97     | [Config](https://github.com/rishabh-mondal/NeurIPS_2025/blob/main/mmrotate_brickkiln/configs/dcfl-le90_r50_1x_dota.py) | [Model](https://drive.google.com/file/d/1Jnr-V3gK9J3Z20u3R0FvCxZBUt3oJcW0/view?usp=drive_link) |
| CSL                  | ECCV-20   | Res50       | OBB     | 16.48       | 0.18     | 9.31   | 8.37      | [Config](https://github.com/rishabh-mondal/NeurIPS_2025/blob/main/mmrotate_brickkiln/configs/rotated-retinanet-rbox-le90_r50_fpn_csl-gaussian_amp-1x_dota.py) | [Model](https://drive.google.com/file/d/15yU8fWolXVYr2J_opp1EP52md_-vGu5c/view?usp=drive_link) |
| Rotated RetinaNet    | ICCV-17   | Res50       | OBB     | 34.37       | 2.81     | 17.27  | 22.55     | [Config](https://github.com/rishabh-mondal/NeurIPS_2025/blob/main/mmrotate_brickkiln/configs/rotated_retinanet_obb_r50_fpn_1x_brickkiln_le90.py) | [Model](https://drive.google.com/file/d/1PXDIminRkZ2kUdhiVxgxnXu4A59DbBze/view?usp=drive_link) |
| Rotated ATSS         | CVPR-20   | Res50       | OBB     | 38.79       | 18.68    | 20.49  | 25.27     | [Config](https://github.com/rishabh-mondal/NeurIPS_2025/blob/main/mmrotate_brickkiln/configs/rotated-atss-le90_r50_fpn_1x_dota.py) | [Model](https://drive.google.com/file/d/1UhXnogbNmree2PYZ5elGPHdie68D_oXb/view?usp=drive_link) |
| GWD                  | ICML-21   | Res50       | OBB     | 41.70       | 0.17     | 22.21  | 25.12     | [Config](https://github.com/rishabh-mondal/NeurIPS_2025/blob/main/mmrotate_brickkiln/configs/rotated-retinanet-rbox-le90_r50_fpn_gwd_1x_dota.py) | [Model](https://drive.google.com/file/d/1UhXnogbNmree2PYZ5elGPHdie68D_oXb/view?usp=drive_link) |
| R³Det                | AAAI-21   | Res50       | OBB     | 43.70       | 0.17     | 24.89  | 28.54     | [Config](https://github.com/rishabh-mondal/NeurIPS_2025/blob/main/mmrotate_brickkiln/configs/r3det_r50_fpn_1x_brickkiln_oc.py) | [Model](https://drive.google.com/file/d/1zBHILdiQ8PeRsjwilFhZd3ZAepL-_5FN/view?usp=drive_link) |
| S²A-Net              | TGRS-21   | Res50       | OBB     | 54.28       | 32.10    | 32.21  | 39.85     | [Config](https://github.com/rishabh-mondal/NeurIPS_2025/blob/main/mmrotate_brickkiln/configs/s2anet-le90_r50_fpn_1x_dota.py) | [Model](https://drive.google.com/file/d/1XMV1EwWHC3M8iuQZuBn6w71m1NLKl986/view?usp=drive_link) |
| ConvNeXt             | CVPR-22   | Res50       | OBB     | 66.19       | 34.33    | 41.84  | 43.53     | [Config](https://github.com/rishabh-mondal/NeurIPS_2025/blob/main/mmrotate_brickkiln/configs/rotated-retinanet-rbox-le90_convnext-tiny_fpn_kld-stable_adamw-1x_dota.py) | [Model](https://drive.google.com/file/d/1709IlyFYj5OGEAh_ZhgU1F-iSgRbxXRI/view?usp=drive_link) |
| YOLOv11L-OBB         | arXiv-24  | CSPDr53     | OBB     | 75.20       | 58.57    | 55.43  | 53.90     | [Config](#) | [Model](#) |
| YOLOv12L             | arXiv-25  | CSPDr53     | AA      | 79.54       | 58.00    | 60.19  | 54.83     | [Config](#) | [Model](#) |
| 🟩 **YOLOv8L-WORLDv2** | CVPR-24 | CSPDr53     | AA      | 🟩 **80.74** | 57.78    | 🟩 **61.28** | 🟩 **56.89** | [Config](#) | [Model](#) |

</details>

---

<details>
<summary><strong>🔷 DETR-Based Models</strong></summary>

| 🏷️ Model       | 📄 Paper  | Backbone | BBox | CA mAP50 | CFCBK | FCBK | Zigzag | ⚙️ Config | 💾 Model |
|----------------|-----------|-------------|---------|-------------|----------|--------|-----------|-----------|----------|
| DETA           | ICCV-23   | Res50       | AA      | 65.34       | 44.01    | 47.56  | 55.21     | [Config](#) | [Model](#) |
| 🟩 **RFDETR**   | arXiv-25  | Dinov2      | AA      | 79.64       | 🟩 **64.26** | 63.68  | 64.25     | [Config](#) | [Model](#) |
| 🟩 **RTDETR**   | CVPR-24   | Res101      | AA      | 🟩 **87.53** | 63.03    | 🟩 **68.60** | 🟩 **64.30** | [Config](#) | [Model](#) |

</details>

---

## 📦 Subprojects

- 🔄 [`mmrotate`](./mmrotate/README.md) — OBB Detection with MMRotate  
- 🔧 [`rfdetr`](./rfdetr/README.md) — DETR-based RFDETR implementation  
- 🚀 [`yolo`](./yolo/README.md) — YOLOv8/YOLOv12 implementation with custom AA head  
- 📁 [`dataset`](./dataset/README.md) — Brick kiln dataset splits, annotations, & pre-processing tools  

---

## 📌 Citation (Coming Soon)

