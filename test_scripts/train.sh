#!/usr/bin/env bash

FILE=$1
CONFIG=$2
GPUS=$3
NODE=${NODE:-1}
NODE_RANK=${NODE_RANK:-0}
ADDR=${ADDR:-127.0.0.1}
PORT=${PORT:-12345}

LOG_FILENAME=test_logs/2n4g
mkdir -p $LOG_FILENAME

export ONEFLOW_FUSE_OPTIMIZER_UPDATE_CAST=true
export ONEFLOW_DEBUG_MODE=1
echo $ONEFLOW_DEBUG_MODE
export SBP_INFER_RULE_TAG=2
echo $SBP_INFER_RULE_TAG
export GLOG_v=3
echo $GLOG_v
#export ONEFLOW_PROFILER_KERNEL_PROFILE_KERNEL_FORWARD_RANGE=True

# --delay=500
#nsys profile --stats true --output ${LOG_FILENAME} --sample none --cpuctxsw none \
python3 -m oneflow.distributed.launch \
--nproc_per_node $GPUS --nnodes $NODE --node_rank $NODE_RANK --master_addr $ADDR --master_port $PORT \
$FILE --config-file $CONFIG ${@:4} \
train.output_dir=$LOG_FILENAME 2>&1| tee ${LOG_FILENAME}/output.log
