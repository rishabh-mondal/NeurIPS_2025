_base_ = [
    '../_base_/datasets/dota.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['projects.OrientedFormer.orientedformer'], allow_failed_imports=False)

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'

num_stages = 2
num_proposals = 100
num_classes = 15
angle_version = 'le90'
depths = [2, 2, 6, 2]

model = dict(
    type='OrientedDDQRCNN',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),
    backbone=dict(
        type='mmdet.SwinTransformer',
        embed_dims=96,
        depths=depths,
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='ChannelMapperWithGN',
        kernel_size=1,
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=4),
    rpn_head=dict(
        type='OrientedAdaMixerDDQ',
        angle_version=angle_version,
        ddq_num_classes=num_classes,
        num_proposals=num_proposals,
        in_channels=256,
        feat_channels=256,
        strides=[4, 8, 16, 32],
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        dqs_cfg=dict(type='nms_rotated', iou_threshold=0.7, nms_pre=1000),
        offset=0.5,
        aux_loss=dict(
            loss_cls=dict(
                type='mmdet.QualityFocalLoss',
                use_sigmoid=True,
                activated=True,
                beta=2.0,
                loss_weight=1.0),
            loss_bbox=dict(type='RotatedIoULoss', mode='linear', loss_weight=5.0),
            train_cfg=dict(
                assigner=dict(
                    type='TopkHungarianAssigner',
                    topk=8,
                    iou_calculator=dict(type='RBboxOverlaps2D'),
                    cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
                    reg_cost=dict(type='RBBoxL1Cost', weight=2.0,
                                  box_format='xywht', angle_version=angle_version),
                    iou_cost=dict(type='RotatedIoUCost', iou_mode='iou', weight=5.0)
                ),
                alpha=1,
                beta=6)),
        main_loss=dict(
            loss_cls=dict(
                type='mmdet.QualityFocalLoss',
                use_sigmoid=True,
                activated=True,
                beta=2.0,
                loss_weight=1.0),
            loss_bbox=dict(type='RotatedIoULoss', mode='linear', loss_weight=5.0),
            train_cfg=dict(
                assigner=dict(
                    type='TopkHungarianAssigner',
                    topk=8,
                    iou_calculator=dict(type='RBboxOverlaps2D'),
                    cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
                    reg_cost=dict(type='RBBoxL1Cost', weight=2.0,
                                  box_format='xywht', angle_version=angle_version),
                    iou_cost=dict(type='RotatedIoUCost', iou_mode='iou', weight=5.0)
                ),
                alpha=1,
                beta=6))),
    roi_head=dict(
        type='OrientedAdaMixerDecoder',
        num_stages=num_stages,
        stage_loss_weights=[1] * num_stages,
        content_dim=256,
        featmap_strides=[4, 8, 16, 32],
        bbox_head=[
            dict(
                type='OrientedFormerDecoderLayer',
                num_classes=num_classes,
                angle_version=angle_version,
                reg_predictor_cfg=dict(type='mmdet.Linear'),
                cls_predictor_cfg=dict(type='mmdet.Linear'),
                num_cls_fcs=1,
                num_reg_fcs=1,
                content_dim=256,
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(1., 1., 1., 1., 1.),
                self_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.0),
                o3d_attn_cfg=dict(
                    type='OrientedAttention',
                    n_points=32,
                    n_heads=64,
                    embed_dims=256,
                    reduction=4),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True)),
                loss_bbox=dict(type='mmdet.L1Loss', loss_weight=2.0),
                loss_iou=dict(type='RotatedIoULoss', mode='linear', loss_weight=5.0),
                loss_cls=dict(
                    type='mmdet.FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                bbox_coder=dict(
                    type='DeltaXYWHTRBBoxCoder')) for _ in range(num_stages)]),
    train_cfg=dict(
        rpn=None,
        rcnn=[
            dict(
                assigner=dict(
                    type='mmdet.HungarianAssigner',
                    match_costs=[
                        dict(type='mmdet.FocalLossCost', weight=2.0),
                        dict(type='RBBoxL1Cost', weight=2.0,
                             box_format='xywht', angle_version=angle_version),
                        dict(type='RotatedIoUCost', iou_mode='iou', weight=5.0)]),
                sampler=dict(type='mmdet.PseudoSampler'),
                pos_weight=1) for _ in range(num_stages)]),
    test_cfg=dict(rpn=None, rcnn=dict(max_per_img=num_proposals)))

# learning rate multipliers
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {
    'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    'backbone.patch_embed.norm': backbone_norm_multi,
    'backbone.norm': backbone_norm_multi,
    'absolute_pos_embed': backbone_embed_multi,
    'relative_position_bias_table': backbone_embed_multi,
    'query_embed': embed_multi,
    'query_feat': embed_multi,
    'level_embed': embed_multi
}
custom_keys.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.downsample.norm': backbone_norm_multi
    for stage_id in range(len(depths) - 1)
})

# optimizer
optim_wrapper = dict(
    optimizer=dict(_delete_=True, type='AdamW', lr=5e-5, weight_decay=1e-6),
    clip_grad=dict(max_norm=1, norm_type=2))

# training schedule
train_cfg = dict(val_interval=6)

# dataset settings
img_scale = (128, 128)
dataset_type = 'DOTADataset'
data_root = '/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/jeet/test/HRDet/data/dota/'

train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/labelTxt/',
        data_prefix=dict(img_path='train/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=[
            dict(type='mmdet.LoadImageFromFile'),
            dict(type='mmdet.LoadAnnotations', with_bbox=True),
            dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
            dict(type='mmdet.RandomFlip', prob=0.5),
            dict(type='mmdet.PackDetInputs')
        ]
    )
)

val_dataloader = dict(
    batch_size=128,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val/labelTxt/',
        data_prefix=dict(img_path='val/images/'),
        test_mode=True,
        pipeline=[
            dict(type='mmdet.LoadImageFromFile'),
            dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
            dict(type='mmdet.LoadAnnotations', with_bbox=True),
            dict(type='mmdet.PackDetInputs')
        ]
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(type='DOTAMetric', metric='mAP')
test_evaluator = val_evaluator

auto_scale_lr = None
num_epochs = 25
runner = dict(type='EpochBasedRunner', max_epochs=num_epochs, val_interval=1)
