
# RF-DETR for Brick Kiln Detection

This repository applies the **Roboflow-DETR (RF-DETR)** object detection model to satellite imagery for **brick kiln detection**.

> Official RF-DETR repository: [Roboflow GitHub](https://github.com/roboflow/rf-detr)

## Results
| Model | # Images | Batch Size | CFCBK | FCBK | Zigzag | mAP50 | Class-Agnostic mAP50 | Model | Logs |
|:-----:|:--------:|:----------:|:-----:|:----:|:------:|:-----:|:--------------------:|:----:|:----:|
| RF-DETR Large | 15,590 | 20 | 64.26 | 63.68 | 64.25 | 64.06 | 79.64 | [model]() | [log](./logs/rfdetr_large/sentinel_data.log) |



<!-- ## Table of Contents

- [Overview](#overview)
- [Dataset Structure](#dataset-structure)
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation) -->

## Overview
Brick kilns contribute significantly to air pollution, and their identification from satellite imagery is essential for environmental monitoring and policy enforcement. This project trains and evaluates RF-DETR on the Sentinel dataset to detect:

- **CFCBK**: Circular Fixed Chimney Brick Kilns  
- **FCBK**: Fixed Chimney Brick Kilns  
- **Zigzag**: Zigzag Brick Kilns



## Dataset Structure

RF-DETR expects datasets in COCO format, organized as follows:

```
data/
├── train/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (other image files)
├── valid/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (other image files)
└── test/
    ├── _annotations.coco.json
    ├── image1.jpg
    ├── image2.jpg
    └── ... (other image files)
```

Each split (`train`, `valid`, `test`) must have its own `_annotations.coco.json` and corresponding images.

DOTA format annotations can be converted to COCO format using the provided [conversion script](./conversion/dota_to_coco.ipynb).

---

>**💡Note:** To customize dataset paths, edit the `build_roboflow` function in `.venv/lib/python3.11/site-packages/rfdetr/datasets/coco.py`:
> ```python
> PATHS = {
>     "train": (root / "train", root / "train" / "_annotations.coco.json"),
>     "val": (root / "valid", root / "valid" / "_annotations.coco.json"),
>     "test": (root / "test", root / "test" / "_annotations.coco.json"),
> }
> ```

---

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/rishabh-mondal/NeurIPS_2025.git
    cd rfdetr
    ```
2. **Create a virtual environment and install dependencies:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```


## Training

To train RF-DETR on your dataset:
```bash
export CUDA_VISIBLE_DEVICES=0
nohup python rfdetr_train.py > ./logs/rfdetr_large/sentinel_data.log 2>&1 &
```

TensorBoard logs can be viewed using:
```bash
tensorboard --logdir ./runs/sentinel_data
```


## Evaluation

For evaluation and analysis, use the provided Jupyter notebooks:
- [rfdetr_eval.ipynb](./rfdetr_eval.ipynb) — mAP calculation using supervision
- [rfdetr_loss.ipynb](./rfdetr_loss.ipynb) — loss curves and convergence analysis



## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements or bug fixes.

For questions or collaborations, please contact [Shardul Junagade](mailto:shardul.junagade@iitgn.ac.in).