set -ex
# bash args_deepspeed_t5.sh 2 8 $MLP_ROLE_INDEX $MLP_WORKER_0_HOST 2 4 true true 80 1280

export OMP_NUM_THREADS=1

# volcengine.com
export NCCL_IB_PCI_RELAXED_ORDERING=1

NNODES=${1:-1}
GPUS_PER_NODE=${2:-8}
# Change for multinode config
NODE_RANK=${3:-0}
MASTER_ADDR=${4:-"127.0.0.1"}
MASTER_PORT=6000
MP=${5:-1}
PP=${6:-1}
USE_FP16=${7:-true}
ACTIVATION_CHECKPOINT=${8:-false}
MICRO_BATCH_SIZE=${9:-4}
GLOBAL_BATCH_SIZE=${10:-4}
NUM_LAYER=${11:-12}
RUN_COMMIT=${12:-"01b1d32"}
TRAIN_ITERS=${13:-220}
LOG_PERIOD=${14:-100}
CHECKPOINT_PATH=${15:-"checkpoints/t5"}
VOCAB_FILE=${16:-"./libai_dataset/bert-base-chinese-vocab.txt"}
DATA_PATH=${17:-"./libai_dataset/loss_compara_content_sentence"} 


SRC_DIR=$(realpath $(dirname $0)/)
TRAN_MODEL="Megatron-Deepspeed_t5"
RUN_TIME=$(date "+%Y%m%d_%H%M%S%N")
LOG_FOLDER=${SRC_DIR}/test_logs/$RUN_COMMIT/${NNODES}n${GPUS_PER_NODE}g
if [[ ! -z "$LOG_FOLDER" ]]; then
    mkdir -p $LOG_FOLDER
fi

AMP_OR="FP32"
if $USE_FP16; then
    AMP_OR="FP16"
fi

LOG_FILENAME=$LOG_FOLDER/${TRAN_MODEL}_nl${NUM_LAYER}_nah12_hs768_${AMP_OR}_ac${ACTIVATION_CHECKPOINT}_mp${MP}_pp${PP}_mb${MICRO_BATCH_SIZE}_gb${GLOBAL_BATCH_SIZE}_${NNODES}n${GPUS_PER_NODE}g_${RUN_TIME}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT"

T5_ARGS="--num-layers $NUM_LAYER \
         --hidden-size 768 \
         --num-attention-heads 12 \
         --kv-channels 64 \
         --ffn-hidden-size 3072 \
         --encoder-seq-length 512 \
         --decoder-seq-length 128 \
         --max-position-embeddings 512 \
         --lr 0.0001 \
         --lr-decay-iters 990000 \
         --train-iters $TRAIN_ITERS \
         --min-lr 0.00001 \
         --lr-warmup-fraction 0.01 \
         --micro-batch-size $MICRO_BATCH_SIZE \
         --global-batch-size $GLOBAL_BATCH_SIZE \
         --vocab-file $VOCAB_FILE \
         --vocab-extra-ids 100 \
         --split 949,50,1 \
         "

OUTPUT_ARGS=" \
    --log-interval $LOG_PERIOD \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    "
# --checkpoint-activations \

DATA_ARGS=" \
    --save $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    "
# --load $CHECKPOINT_PATH \


ZERO_STAGE=1
DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${ZERO_STAGE} \
    --deepspeed-activation-checkpointing \
    "

export LOGLEVEL=WARNING
# LAUNCHER="deepspeed -num_gpus $GPUS_PER_NODE"

CMD="python3 -m torch.distributed.launch $DISTRIBUTED_ARGS ./pretrain_t5.py \
    $T5_ARGS \
    $OUTPUT_ARGS \
    $DATA_ARGS \
    $DEEPSPEED_ARGS \
    --tensor-model-parallel-size $MP \
    --pipeline-model-parallel-size $PP \
    --DDP-impl local "

if $USE_FP16; then
    CMD+=" \
      --fp16 "
fi

if $ACTIVATION_CHECKPOINT; then
    CMD+=" \
      --checkpoint-activations "
fi

echo "Rum cmd ${CMD}"

$CMD 2>&1 | tee ${LOG_FILENAME}.log
