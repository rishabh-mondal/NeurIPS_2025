#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29606}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# Export visible GPUs
export CUDA_VISIBLE_DEVICES=$GPUS
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# Count number of GPUs
NUM_GPUS=$(echo $GPUS | tr ',' '\n' | wc -l)

# Set Python path and launch training using torch.distributed
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

LOG_DIR="train_$(date +%Y%m%d_%H%M%S).log"

nohup python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --seed 0 \
    --launcher pytorch ${@:3} > $LOG_DIR 2>&1 &

echo "Training started. Logs are being written to $LOG_DIR"




# conda activate open-mmlab
# cd mmrotate

# bash tools/dist_train.sh 'configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_brickkiln_le90.py' 3

# bash tools/dist_train.sh 'configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_brickkiln_le90.py' 1

# bash tools/dist_train.sh 'configs/redet/redet_re50_refpn_1x_brickkiln_le90.py' 3

# bash tools/dist_train.sh 'configs/roi_trans/roi_trans_swin_tiny_fpn_1x_brickkiln_le90.py' 2

# bash tools/dist_train.sh 'configs/s2anet/s2anet_r50_fpn_1x_brickkiln_le135.py' 3

# bash tools/dist_train.sh 'configs/gliding_vertex/gliding_vertex_r50_fpn_1x_brickkiln_le90.py' 3

# bash tools/dist_train.sh 'configs/r3det/r3det_r50_fpn_1x_brickkiln_oc.py' 3

# bash tools/dist_train.sh 'configs/r3det/r3det_tiny_r50_fpn_1x_brickkiln_oc.py' 3


# bash tools/dist_train.sh 'configs/roi_trans/roi_trans_swin_tiny_fpn_1x_up_le90.py' 0

# bash tools/dist_train.sh 'configs/s2anet/s2anet_r50_fpn_1x_up_le135.py' 3