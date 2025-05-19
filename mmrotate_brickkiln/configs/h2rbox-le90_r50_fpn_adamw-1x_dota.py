angle_version = 'le90'
backend_args = None
data = dict(samples_per_gpu=128)
data_root = '/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/jeet/test/HRDet/data/dota/'
dataset_type = 'DOTADataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
default_scope = 'mmrotate'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='mmdet.ResNet'),
    bbox_head=dict(
        angle_version='le90',
        bbox_coder=dict(angle_version='le90', type='DistanceAnglePointCoder'),
        center_sample_radius=1.5,
        center_sampling=True,
        centerness_on_reg=True,
        crop_size=(
            128,
            128,
        ),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='mmdet.IoULoss'),
        loss_bbox_ss=dict(
            angle_loss_cfg=dict(loss_weight=1.0, type='mmdet.L1Loss'),
            center_loss_cfg=dict(loss_weight=0.0, type='mmdet.L1Loss'),
            loss_weight=0.4,
            shape_loss_cfg=dict(loss_weight=1.0, type='mmdet.IoULoss'),
            type='H2RBoxConsistencyLoss'),
        loss_centerness=dict(
            loss_weight=1.0, type='mmdet.CrossEntropyLoss', use_sigmoid=True),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='mmdet.FocalLoss',
            use_sigmoid=True),
        norm_on_bbox=True,
        num_classes=3,
        scale_angle=True,
        square_classes=[
            9,
            11,
        ],
        stacked_convs=4,
        strides=[
            8,
            16,
            32,
            64,
            128,
        ],
        type='H2RBoxHead',
        use_hbbox_loss=False),
    crop_size=(
        128,
        128,
    ),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        boxtype2tensor=False,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='mmdet.DetDataPreprocessor'),
    neck=dict(
        add_extra_convs='on_output',
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        relu_before_extra_convs=True,
        start_level=1,
        type='mmdet.FPN'),
    test_cfg=dict(
        max_per_img=2000,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.1, type='nms_rotated'),
        nms_pre=2000,
        score_thr=0.05),
    train_cfg=None,
    type='H2RBoxDetector')
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0001, type='AdamW', weight_decay=0.05),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=500,
        start_factor=0.3333333333333333,
        type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=20,
        gamma=0.1,
        milestones=[
            8,
            11,
        ],
        type='MultiStepLR'),
]
resume = False
runner = dict(max_epochs=25)
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=128,
    dataset=dict(
        ann_file='val/labelTxt/',
        data_prefix=dict(img_path='val/images/'),
        data_root=
        '/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/jeet/test/HRDet/data/dota/',
        pipeline=[
            dict(backend_args=None, type='mmdet.LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                128,
                128,
            ), type='mmdet.Resize'),
            dict(
                box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='ConvertBoxType'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='DOTADataset'),
    drop_last=False,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(metric='mAP', type='DOTAMetric')
test_pipeline = [
    dict(backend_args=None, type='mmdet.LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        128,
        128,
    ), type='mmdet.Resize'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='mmdet.PackDetInputs'),
]
train_cfg = dict(max_epochs=25, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=None,
    batch_size=128,
    dataset=dict(
        ann_file='train/labelTxt/',
        data_prefix=dict(img_path='train/images/'),
        data_root=
        '/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/jeet/test/HRDet/data/dota/',
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=[
            dict(backend_args=None, type='mmdet.LoadImageFromFile'),
            dict(
                box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                box_type_mapping=dict(gt_bboxes='hbox'),
                type='ConvertBoxType'),
            dict(
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='ConvertBoxType'),
            dict(keep_ratio=True, scale=(
                128,
                128,
            ), type='mmdet.Resize'),
            dict(
                direction=[
                    'horizontal',
                    'vertical',
                    'diagonal',
                ],
                prob=0.75,
                type='mmdet.RandomFlip'),
            dict(type='mmdet.PackDetInputs'),
        ],
        type='DOTADataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='mmdet.LoadImageFromFile'),
    dict(box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
    dict(box_type_mapping=dict(gt_bboxes='hbox'), type='ConvertBoxType'),
    dict(box_type_mapping=dict(gt_bboxes='rbox'), type='ConvertBoxType'),
    dict(keep_ratio=True, scale=(
        128,
        128,
    ), type='mmdet.Resize'),
    dict(
        direction=[
            'horizontal',
            'vertical',
            'diagonal',
        ],
        prob=0.75,
        type='mmdet.RandomFlip'),
    dict(type='mmdet.PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=128,
    dataset=dict(
        ann_file='val/labelTxt/',
        data_prefix=dict(img_path='val/images/'),
        data_root=
        '/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/jeet/test/HRDet/data/dota/',
        pipeline=[
            dict(backend_args=None, type='mmdet.LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                128,
                128,
            ), type='mmdet.Resize'),
            dict(
                box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='ConvertBoxType'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='DOTADataset'),
    drop_last=False,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(metric='mAP', type='DOTAMetric')
val_pipeline = [
    dict(backend_args=None, type='mmdet.LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        128,
        128,
    ), type='mmdet.Resize'),
    dict(box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
    dict(box_type_mapping=dict(gt_bboxes='rbox'), type='ConvertBoxType'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='mmdet.PackDetInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='RotLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/h2rbox'
