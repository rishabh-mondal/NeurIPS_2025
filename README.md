# SentinelKilnDB: A Large-Scale Dataset and Benchmark for OBB Brick Kiln Detection in South Asia Using Satellite Imagery [NeurIPS 2025]

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
