# 🧱 SentinelKilnDB: A Large-Scale Dataset and Benchmark for OBB Brick Kiln Detection in South Asia Using Satellite Imagery  
**[NeurIPS 2025 Submission]**

---

## 📊 Results and Benchmarks

> 🟩 **Highest score** per column is highlighted  
> 🔽 Click on a category to expand its models  
> 🏷️ *Badges* indicate model families for quick reference

---

<details>
<summary><strong>🧠 Two-Stage Models</strong></summary>

| 🏷️ Model       | 📄 Paper   | 🧠 Backbone | BBox | 🎯 CA mAP50 | 🧱 CFCBK | 🧱 FCBK | 🧱 Zigzag | ⚙️ Config | 💾 Model |
|:---------------|:----------:|:-----------:|:-------:|:-----------:|:--------:|:------:|:--------:|:---------:|:--------:|
| PSC            | CVPR-23    | Res50       | OBB     | 27.41       | 0.38     | 13.83  | 17.03    | | [📄 PSC Dual AMP Config](https://github.com/rishabh-mondal/NeurIPS_2025/blob/main/configs/psc/rotated-retinanet-rbox-le90_r50_fpn_psc-dual_amp-1x_dota.py)
 |[model]https://drive.google.com/file/d/1yGB0_fcGndLI9uCf678tE3OiHAYi3jds/view?usp=drive_link|
| H2RBox         | ICLR-23    | Res50       | OBB     | 47.01       | 24.93    | 30.27  | 31.02    | [cfg](#)  | [model](#) |
| **RoI Transformer** | CVPR-19 | Swin-T      | OBB     | 🟩 **61.65** | 🟩 **45.31** | 🟩 **43.75** | 🟩 **47.46** | [cfg](#) | [model](#) |

</details>

---

<details>
<summary><strong>⚡ One-Stage Models</strong></summary>

| 🏷️ Model           | 📄 Paper   | 🧠 Backbone | 📦 BBox | 🎯 CA mAP50 | 🧱 CFCBK | 🧱 FCBK | 🧱 Zigzag | ⚙️ Config | 💾 Model |
|:-------------------|:----------:|:-----------:|:-------:|:-----------:|:--------:|:------:|:--------:|:---------:|:--------:|
| Rotated FCOS       | ICCV-19    | Res50       | OBB     | 15.62       | 12.72    | 8.48   | 9.99     | [cfg](#)  | [model](#) |
| DCFL               | CVPR-23    | Res50       | OBB     | 16.01       | 9.07     | 6.51   | 12.97    | [cfg](#)  | [model](#) |
| CSL                | ECCV-20    | Res50       | OBB     | 16.48       | 0.18     | 9.31   | 8.37     | [📄 CSL Gaussian Config](https://github.com/rishabh-mondal/NeurIPS_2025/blob/main/mmrotate_brickkiln/configs/rotated-retinanet-rbox-le90_r50_fpn_csl-gaussian_amp-1x_dota.py)  | [model](https://drive.google.com/file/d/15yU8fWolXVYr2J_opp1EP52md_-vGu5c/view?usp=drive_link) |
| Rotated RetinaNet  | ICCV-17    | Res50       | OBB     | 34.37       | 2.81     | 17.27  | 22.55    | [cfg](#)  | [model](#) |
| Rotated ATSS       | CVPR-20    | Res50       | OBB     | 38.79       | 18.68    | 20.49  | 25.27    | [cfg](#)  | [model](#) |
| GWD                | ICML-21    | Res50       | OBB     | 41.70       | 0.17     | 22.21  | 25.12    | [cfg](#)  | [model](#) |
| R³Det              | AAAI-21    | Res50       | OBB     | 43.70       | 0.17     | 24.89  | 28.54    | [cfg](#)  | [model](#) |
| S²A-Net            | TGRS-21    | Res50       | OBB     | 54.28       | 32.10    | 32.21  | 39.85    | [cfg](#)  | [model](#) |
| ConvNeXt           | CVPR-22    | Res50       | OBB     | 66.19       | 34.33    | 41.84  | 43.53    | [cfg](#)  | [model](#) |
| YOLOv11L-OBB       | arXiv-24   | CSPDr53     | OBB     | 75.20       | 58.57    | 55.43  | 53.90    | [cfg](#)  | [model](#) |
| YOLOv12L           | arXiv-25   | CSPDr53     | AA      | 79.54       | 58.00    | 60.19  | 54.83    | [cfg](#)  | [model](#) |
| **YOLOv8L-WORLDv2**| CVPR-24    | CSPDr53     | AA      | 🟩 **80.74** | 57.78    | 🟩 **61.28** | 🟩 **56.89** | [cfg](#) | [model](#) |

</details>

---

<details>
<summary><strong>🔷 DETR-Based Models</strong></summary>

| 🏷️ Model  | 📄 Paper   | 🧠 Backbone | 📦 BBox | 🎯 CA mAP50 | 🧱 CFCBK | 🧱 FCBK | 🧱 Zigzag | ⚙️ Config | 💾 Model |
|:----------|:----------:|:-----------:|:-------:|:-----------:|:--------:|:------:|:--------:|:---------:|:--------:|
| DETA      | ICCV-23    | Res50       | AA      | 65.34       | 44.01    | 47.56  | 55.21    | [cfg](#)  | [model](#) |
| RFDETR    | arXiv-25   | Dinov2      | AA      | 79.64       | 🟩 **64.26** | 63.68  | 64.25    | [cfg](#)  | [model](#) |
| **RTDETR**| CVPR-24    | Res101      | AA      | 🟩 **87.53** | 63.03    | 🟩 **68.60** | 🟩 **64.30** | [cfg](#) | [model](#) |

</details>

---

## 📦 Subprojects

- 🔄 [`mmrotate`](./mmrotate/README.md) — OBB Detection with MMRotate  
- 🔧 [`rfdetr`](./rfdetr/README.md) — DETR-based detection with RFDETR

---

## 📈 Interactive Plots (Coming Soon)

We'll provide interactive charts and analytics comparing:

- mAP trends across models
- Class-specific AP
- Runtime vs. accuracy tradeoffs

> Powered by Plotly / Altair visual dashboards

---

## 🧰 Setup

See subproject READMEs linked above for:
- Installation
- Training
- Evaluation
- Model conversion
