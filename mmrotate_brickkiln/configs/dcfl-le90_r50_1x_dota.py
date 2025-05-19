_base_ = [
    '../_base_/datasets/dota.py',  # Update path if needed
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

angle_version = 'le90'
num_epochs = 25

model = dict(
    type='mmdet.RetinaNet',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='mmdet.RetinaHead',
        num_classes=3,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='FakeRotatedAnchorGenerator',
            angle_version=angle_version,
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[1.0, 0.5, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHTRBBoxCoder',
            angle_version=angle_version,
            norm_factor=None,
            edge_swap=True,
            proj_xy=True,
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.L1Loss',
            loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='mmdet.MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='RBboxOverlaps2D')),
        sampler=dict(
            type='mmdet.PseudoSampler'),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000)
)

# Image normalization
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

# Pipeline for training
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True),
    dict(type='mmdet.Resize', img_scale=(128, 128), keep_ratio=True),
    dict(type='mmdet.RandomFlip', flip_ratio=0.5),
    dict(type='mmdet.Normalize', **img_norm_cfg),
    dict(type='mmdet.Pad', size_divisor=32),
    dict(type='mmdet.DefaultFormatBundle'),
    dict(type='mmdet.Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

# Dataset definition
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=8,
    train=dict(pipeline=train_pipeline, version=angle_version),
    val=dict(version=angle_version),
    test=dict(version=angle_version)
)

# Optimizer
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)

# Learning policy
auto_scale_lr = dict(enable=True, base_batch_size=128)

# Runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=num_epochs)

# Checkpoint saving and evaluation frequency
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric='mAP')
