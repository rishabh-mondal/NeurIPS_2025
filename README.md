<div align="center">

# 🛰️ **SentinelKilnDB**  
### *A Large-Scale Dataset and Benchmark for Oriented Bounding Box (OBB) Brick Kiln Detection in South Asia Using Satellite Imagery*  

**Conference:**  🎓 *NeurIPS 2025 – Datasets & Benchmarks Track*

---

<a href="https://neurips.cc/virtual/2025/poster/121530" target="_blank">
  <img src="https://img.shields.io/badge/Read%20Paper-2ecc71?style=for-the-badge&logo=readme&logoColor=white" alt="Read Paper Button">
</a>
&nbsp;&nbsp;
<a href="https://sustainability-lab.github.io/sentinelkilndb/" target="_blank">
  <img src="https://img.shields.io/badge/Project%20Page-1e90ff?style=for-the-badge&logo=github&logoColor=white" alt="Project Page Button">
</a>
&nbsp;&nbsp;
<a href="https://www.kaggle.com/datasets/rishabhsnip/sentinelkiln-dataset" target="_blank">
  <img src="https://img.shields.io/badge/Dataset-f39c12?style=for-the-badge&logo=kaggle&logoColor=white" alt="Dataset Button">
</a>

---

</div>

### Overview

**SentinelKilnDB** is a comprehensive benchmark dataset for detecting **brick kilns**—major unorganized emission sources—across **South Asia** using **Sentinel-2 multispectral satellite imagery**.  
It provides oriented bounding box (OBB) annotations for diverse kiln types (CFCBK, FCBK, Zigzag), enabling robust and policy-relevant model development for large-scale environmental monitoring.

---

<div align="center">

## 🗺️ **Dataset Overview — SentinelKilnDB**

</div>

<table align="center">
<tr>
<th align="left">📂 Attribute</th>
<th align="left">🧭 Details</th>
</tr>

<tr><td><b>Dataset</b></td><td>SentinelKilnDB</td></tr>
<tr><td><b>Size</b></td><td>~4.00 GB</td></tr>
<tr><td><b>Images</b></td><td>114,300 RGB tiles (10 m resolution)</td></tr>
<tr><td><b>Satellite</b></td><td>Sentinel-2 (MSI)</td></tr>
<tr><td><b>Time Period</b></td><td>September 2023 – February 2024</td></tr>
<tr><td><b>Annotations</b></td><td>97,648 Oriented Bounding Boxes (OBBs)</td></tr>
<tr><td><b>Classes</b></td><td>3 – CFCBK, FCBK, Zigzag</td></tr>
<tr><td><b>Format</b></td><td>DOTA / YOLO-OBB / YOLO-AA</td></tr>
<tr><td><b>License</b></td><td>CC BY-NC-SA 4.0</td></tr>
</table>

---

<!-- <div align="center">
<a href="https://www.kaggle.com/datasets/rishabhsnip/sentinelkiln-dataset" target="_blank">
  <img src="https://img.shields.io/badge/Download%20Dataset-f39c12?style=for-the-badge&logo=kaggle&logoColor=white" alt="Download Dataset">
</a>
</div> -->

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

Class wise stratified split for balanced representation.

| Split | Images (.png) | Label files (.txt) | No. of BBoxes |
|-------|---------------:|-------------------:|---------------:|
| Train | 71,856 | 47,214 | 63,787 |
| Val | 23,952 | 15,738 | 21,042 |
| Test | 18,492 | 10,278 | 12,819 |
| **Total** | **114,300** | **73,239** | **97,648** |


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

> **Tip**: Each script is modular and can be executed independently, making the pipeline flexible and customizable.

---

# Unified Setup Guide — One-Cell Environment

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
### Model Training Commands

| Model           | Steps                                                                                       | Commands                                                                                                          |
|-----------------|---------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| **MMRotate**    | 1. Clone repo and copy configs to `mmrotate` folder<br>2. Run distributed training on GPUs  | ```bash<br>git clone https://github.com/rishabh-mondal/NeurIPS_2025.git<br>cp -r NeurIPS_2025/mmrotate/mmrotate_brickkiln/* mmrotate/<br>cd mmrotate<br>bash tools/dist_train.sh configs/rotated-retinanet-rbox-le90_r50_fpn_psc-dual_amp-1x_dota.py 0,1,2,3<br>``` |
| **Ultralytics** | Run training script                                                                          | ```bash<br>cd training_scripts<br>chmod +x train.sh<br>./train.sh<br>```                                          |
| **RFDETR**      | Run training with CUDA device 0 and save logs                                              | ```bash<br>cd training_scripts<br>export CUDA_VISIBLE_DEVICES=0<br>nohup python rfdetr_train.py > ./logs/rfdetr_large/sentinel_data.log 2>&1 &<br>``` |
| **DETA**        | Run training with CUDA device 0 and save logs                                              | ```bash<br>cd training_scripts<br>export CUDA_VISIBLE_DEVICES=0<br>nohup python deta_train.py > ./logs/deta_large/sentinel_data.log 2>&1 &<br>```       |


---

## 🔍 Model Inference and Evaluation

| Model         | Task        | Command / Notebook                                                                                   |
|---------------|-------------|-------------------------------------------------------------------------------------------------------|
| **MMRotate**  | Inference   | ```bash<br>cd model_inference_evaluation_scripts<br>export CUDA_VISIBLE_DEVICES=0<br>python folder_inference_mine.py<br>``` |
|               | Evaluation  | [mmrotate_eval.ipynb](./mmrotate_eval.ipynb)                                                         |
| **Ultralytics** | Evaluation | ```bash<br>cd model_inference_evaluation_scripts<br>export CUDA_VISIBLE_DEVICES=0<br>```<br>[ultralytics_eval.ipynb](./ultralytics_eval.ipynb) |
| **RFDETR**    | Inference   | ```bash<br>cd model_inference_evaluation_scripts<br>export CUDA_VISIBLE_DEVICES=0<br>python rfdetr_inference.py<br>``` |
|               | Evaluation  | ```bash<br>python rfdetr_eval.py<br>```                                                              |
| **DETA**      | Inference   | ```bash<br>cd model_inference_evaluation_scripts<br>export CUDA_VISIBLE_DEVICES=0<br>python deta_inference.py<br>``` |


---

## Model Performance


> 🟩 **Highest score** per column is highlighted  
> 🔽 Click on a category to expand its models  
> 🏷️ *Badges* indicate model families for quick reference

---

<details>
<summary><strong>Two-Stage Models</strong></summary>

| 🏷️ Model             | 📄 Paper  | Backbone | BBox | CA mAP50 | CFCBK | FCBK | Zigzag | ⚙️ Config | 💾 Model |
|----------------------|-----------|-------------|---------|-------------|----------|--------|-----------|-----------|----------|
| PSC                  | CVPR-23   | Res50       | OBB     | 27.41       | 0.38     | 13.83  | 17.03     | [📄 Config](https://github.com/rishabh-mondal/NeurIPS_2025/blob/main/configs/psc/rotated-retinanet-rbox-le90_r50_fpn_psc-dual_amp-1x_dota.py) | [🔗 Model](https://drive.google.com/file/d/1yGB0_fcGndLI9uCf678tE3OiHAYi3jds/view?usp=drive_link) |
| H2RBox               | ICLR-23   | Res50       | OBB     | 47.01       | 24.93    | 30.27  | 31.02     | [Config](https://github.com/rishabh-mondal/NeurIPS_2025/blob/main/mmrotate_brickkiln/configs/h2rbox-le90_r50_fpn_adamw-1x_dota.py) | [Model](https://drive.google.com/file/d/18Flaofo6yeTioR_1Xo6T7Hnfh94al9f9/view?usp=drive_link) |
| 🟩 **RoI Transformer** | CVPR-19 | Swin-T      | OBB     | 🟩 **61.65** | 🟩 **45.31** | 🟩 **43.75** | 🟩 **47.46** | [Config](https://github.com/rishabh-mondal/NeurIPS_2025/blob/main/mmrotate_brickkiln/configs/roi_trans_r50_fpn_1x_brickkiln_le90.py) | [Model](https://drive.google.com/file/d/1suX01Id1UR6QD19EZtx8WcM5jKgNLfGq/view?usp=drive_link) |

</details>

---

<details>
<summary><strong>One-Stage Models</strong></summary>

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
<summary><strong>DETR-Based Models</strong></summary>

| 🏷️ Model       | 📄 Paper  | Backbone | BBox | CA mAP50 | CFCBK | FCBK | Zigzag | ⚙️ Config | 💾 Model |
|----------------|-----------|-------------|---------|-------------|----------|--------|-----------|-----------|----------|
| DETA           | ICCV-23   | Res50       | AA      | 65.34       | 44.01    | 47.56  | 55.21     | [Config](#) | [Model](#) |
| 🟩 **RFDETR**   | arXiv-25  | Dinov2      | AA      | 79.64       | 🟩 **64.26** | 63.68  | 64.25     | [Config](#) | [Model](#) |
| 🟩 **RTDETR**   | CVPR-24   | Res101      | AA      | 🟩 **87.53** | 63.03    | 🟩 **68.60** | 🟩 **64.30** | [Config](#) | [Model](#) |

</details>
<details>
<summary><strong>Foundation Models</strong></summary>

| Model       | Paper    | Backbone | BBox | CA mAP50 | CFCBK | FCBK  | Zigzag | Config | Model |
|-------------|----------|----------|------|----------|-------|-------|--------|--------|-------|
| SatMAE++    | CVPR 2024| ViT-L    | AA   | 36.63    | 27.43 | 21.01 | 29.74  | [Config](https://github.com/rishabh-mondal/NeurIPS_2025/tree/main/foundation_model/satmae++) | [Model](https://iitgnacin-my.sharepoint.com/:u:/g/personal/23310002_iitgn_ac_in/EelwlnYCyqdNumi0bFEDyusBVDoyWTSfCTOCFVrHc55nqg?e=09MwA5) |
| CROMA       | NeurIPS 2023 | ViT-B | AA   | 63.89    | 17.69 | 44.66 | 56.40  | [Config](https://github.com/rishabh-mondal/NeurIPS_2025/tree/main/foundation_model/croma) | [Model](https://iitgnacin-my.sharepoint.com/:u:/g/personal/23310002_iitgn_ac_in/EbjNCZt_0-BFhXvsZyqYcMYBtuARb2uGhFZYIdY_hDRS7Q?e=S8Vn6b) |
| Prithvi     | arXiv 2023| ViT-L   | AA   | 59.26    | 16.86 | 40.63 | 52.77  | [Config](https://github.com/rishabh-mondal/NeurIPS_2025/tree/main/foundation_model/prithvi) | [Model](https://iitgnacin-my.sharepoint.com/:u:/g/personal/23310002_iitgn_ac_in/EW7PCdMxTXtKnLOmEQWX9eMBugbZfVbwTEKjFiLZGRJbDA?e=pasBEG) |
| Panopticon  | CVPR 2025| ViT-B    | AA   | 74.87    | 43.13 | 50.61 | 55.11  | [Config](https://github.com/rishabh-mondal/NeurIPS_2025/tree/main/foundation_model/panopticon/) | [Model](https://iitgnacin-my.sharepoint.com/:u:/g/personal/23310002_iitgn_ac_in/EdRmDub80BFDqNRFufxfuBsBXFR4DRqD2Uh7bL6UM6dS5A?e=JYIlLh) |
| SatMAE      | NeurIPS 2022 | ViT-L| AA   | 76.36    | 40.65 | 50.72 | 56.03  | [Config](https://github.com/rishabh-mondal/NeurIPS_2025/tree/main/foundation_model/satmae) | [Model](https://iitgnacin-my.sharepoint.com/:u:/g/personal/23310002_iitgn_ac_in/Efg4C-N1gttMkN02WeTeRtABCvkQB7mh7tfodyBzCXpvHw?e=BfVYit) |
| Copernicus  | ICCV 2025| ViT-B    | AA   | 77.22    | 61.09 | 59.73 | 67.78  | [Config](https://github.com/rishabh-mondal/NeurIPS_2025/tree/main/foundation_model/copernicus) | [Model](https://iitgnacin-my.sharepoint.com/:u:/g/personal/23310002_iitgn_ac_in/Ea0J6W6rMEdAhm3NfG5mZXIBDpgFAtfhBTxaY580bstxwg?e=mNmSq4) |
| Scale MAE   | ICCV 2023| ViT-L    | AA   | 78.43    | 52.10 | 60.77 | 65.18  | [Config](https://github.com/rishabh-mondal/NeurIPS_2025/tree/main/foundation_model/scalemae) | [Model](https://iitgnacin-my.sharepoint.com/:u:/g/personal/23310002_iitgn_ac_in/EQfBet9wS9hFmpMjTMABvQEBZThps8B_8w_M9tjSzX19kA?e=qRuPBI) |
| Galileo     | ICML 2025| ViT-B    | AA   | 86.66    | 72.02 | 69.81 | 72.19  | [Config](https://github.com/rishabh-mondal/NeurIPS_2025/tree/main/foundation_model/galileo) | [Model](https://iitgnacin-my.sharepoint.com/:u:/g/personal/23310002_iitgn_ac_in/EdgqIJyAerVMvqXkfE2e-woB-RMN16nYuJQhFKUzeTXmog?e=PMKR7E) |
| TerraMind   | ICCV 2025| ViT-B    | AA   | 86.91    | 69.04 | 70.54 | 75.55  | [Config](https://github.com/rishabh-mondal/NeurIPS_2025/tree/main/foundation_model/terramind) | [Model](https://iitgnacin-my.sharepoint.com/:u:/g/personal/23310002_iitgn_ac_in/ETCfRDjM-7tCnCtFtKxoXmsBm5-e8bTkw1O0V1v3nzIZRg?e=79N2bh) |

</details>



## Citation 

@inproceedings{mondal2025sentinelkilndb, title={SentinelKilnDB: A Large-Scale Dataset and Benchmark for OBB Brick Kiln Detection in South Asia Using Satellite Imagery}, author={Rishabh Mondal and Jeet Parab and Heer Kubadia and Shataxi Dubey and Shardul Junagade and Zeel B. Patel and Nipun Batra}, booktitle={The Thirty-Ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track}, year={2025}, url={https://openreview.net/forum?id=efGzsxVSEC}

}


## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements or bug fixes.
For questions or collaborations, please contact [Rishabh Mondal](mailto:rishabh.mondal@iitgn.ac.in).