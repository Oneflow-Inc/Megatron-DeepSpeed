## 复现Megatron-Deepspeed的流程
### 准备环境

首先准备多机测试环境 参考

[OneFlow性能测试前的标准流程](https://github.com/Oneflow-Inc/OneTeam/issues/478)

[Megatron-LM多机训练](https://github.com/Oneflow-Inc/OneTeam/issues/328#issuecomment-820375669)

```bash
## 参考 https://github.com/bigscience-workshop/Megatron-DeepSpeed

python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 克隆仓库
git clone https://github.com/bigscience-workshop/Megatron-DeepSpeed.git
cd Megatron-DeepSpeed

# 安装apex
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# 安装deepspeed
cd ..
git clone https://github.com/microsoft/deepspeed
cd deepspeed
rm -rf build
pip install -e . --global-option="build_ext" --global-option="-j8" --no-cache -v --disable-pip-version-check
cd ..

pip install transformers
```
`vim megatron/training.py` 

L798 添加吞吐输出
```python
log_string += ' tpt: {:.1f} samples/s |'.format(batch_size / elapsed_time_per_iteration)
```
L895 添加显存输出
```python
import os
while iteration < args.train_iters:
    if iteration == 101:
        cmd = "nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv"
        os.system(cmd)
```

`vim megatron/model/fused_softmax.py` 将assert mask is None注释掉

### 准备数据集
- 如果对齐libai的数据集
    ```bash
    mkdir libai_dataset && cd libai_dataset && wget https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/bert-base-chinese-vocab.txt && wget https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/gpt2-merges.txt && wget https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/gpt2-vocab.json && wget https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.bin && wget https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.idx && cd ..

- 如果用官方数据集（详见https://github.com/bigscience-workshop/Megatron-DeepSpeed），以GPT为例：
    ```bash
    wget https://huggingface.co/bigscience/misc-test-data/resolve/main/stas/oscar-1GB.jsonl.xz
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
    xz -d oscar-1GB.jsonl.xz
    python tools/preprocess_data.py \
        --input oscar-1GB.jsonl \
        --output-prefix my-gpt2 \
        --vocab gpt2-vocab.json \
        --dataset-impl mmap \
        --tokenizer-type GPT2BPETokenizer \
        --merge-file gpt2-merges.txt \
        --append-eod \
        --workers 8
    ```


### 运行测试
将本仓库test_scripts路径下的 `args_deepspeed_gpt.sh`,  `args_deepspeed_t5.sh` 拷贝到 Megatron-DeepSpeed 仓库下

如果没用到流水并行，则需要注释掉 `DEEPSPEED_ARGS` 中关于deepspeed的三行

如果也没有用到zero，则需要注释掉CMD中的 `$DEEPSPEED_ARGS`

- gpt
```bash
## 以类脑008 009为例
# 用例1 2机分别运行
bash args_deepspeed_gpt.sh 2 8 0 "10.10.0.8" 2 4 true true 24 384
bash args_deepspeed_gpt.sh 2 8 1 "10.10.0.8" 2 4 true true 24 384

# 用例2 2机分别运行
bash args_deepspeed_gpt.sh 2 8 0 "10.10.0.8" 2 4 true true 24 768
bash args_deepspeed_gpt.sh 2 8 1 "10.10.0.8" 2 4 true true 24 768
```

- t5
```bash
bash args_deepspeed_t5.sh 2 4 0 "11.11.1.25" 2 1 true true 16 512
bash args_deepspeed_t5.sh 2 4 1 "11.11.1.25" 2 1 true true 16 512
```


## 复现LiBai+Zero的流程
### 环境准备
```bash
git clone https://github.com/Oneflow-Inc/libai.git
cd libai
python3 -m pip install https://oneflow-staging.oss-cn-beijing.aliyuncs.com/canary/commit/55b822e4d3c88757d11077d7546981309125c73f/cu112/oneflow-0.8.0%2Bcu112.git.55b822e4-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

pip install pybind11
pip install -e .
```
`vim libai/engine/default.py` 注释掉模型保存
```python
    ret = [
        hooks.IterationTimer(),
        hooks.LRScheduler(),  # for beauty lr scheduler printer in `nn.Graph` mode
        #hooks.PeriodicCheckpointer(self.checkpointer, self.cfg.train.checkpointer.period),
    ]
```
`vim libai/engine/trainer.py` 添加显存输出
```python
import os
class TrainerBase:
    def train(self, start_iter: int, max_iter: int):
        with EventStorage(self.start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    if self.iter == 99:
                        cmd = "nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv"
                        os.system(cmd)
                    self.before_step()
                    self.run_step()
                    self.after_step()
```
### 运行测试
- 将 https://github.com/Oneflow-Inc/OneAutoTest/ 仓库libai路径下的 `args_libai_gpt2.sh` 和 `args_libai_t5.sh` 拷贝至libai仓库的tools路径下，CMD考虑是否增加zero相关的两行
- 注：在有模型并行或流水并行时，默认开启zero2，则仅通过修改stage=1来实现zero1是不行的，还需要注释掉`flow.boxing.nccl.enable_use_compute_stream(True)`，详情见 https://github.com/Oneflow-Inc/OneTeam/issues/1435#issuecomment-1219009112 另外，仅在纯数据并行且不带Acc的情况下，zero1才是一个正向优化 
- 将 https://github.com/Oneflow-Inc/OneAutoTest/ 仓库libai路径下的 `gpt2_nl24_nah16_hs1024.py` 和 `t5_nl12_nah12_hs768.py` 拷贝至libai仓库的configs路径下，并修改数据集路径
- 若想对比 LiBai的pipeline_stage_id优化 开启和关闭后的性能，则在 configs/gpt2_nl24_nah16_hs1024.py 中添加和注释掉如下行即可
`train.dist.custom_pipeline_stage_id =  [0] * 6 + [1] * 6 + [2] * 6 + [3] * 6`
- gpt
    ```bash
    # 用例1 2机分别运行
    bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 2 8 0 "10.10.0.8" 2 4 true true 24 384
    bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 2 8 1 "10.10.0.8" 2 4 true true 24 384

    # 用例2 2机分别运行
    bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 2 8 0 "10.10.0.8" 2 4 true true 24 768
    bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 2 8 1 "10.10.0.8" 2 4 true true 24 768
    ```
- projects/T5

    调换libai仓库中的mt5_pretrain.py，改为当前页面下的mt5_pretrain，这个配置是和megatron官方对齐的。libai用main分支测就可以。

    如果想在控制台log里输出 `build model time`，则 `vim libai/engine/default.py` 在 `self.model = self.build_model(cfg)` 前后，修改成：
    ```python
    s = time.time()
    self.model = self.build_model(cfg)
    e = time.time()
    logger.info("build model time ------------------------------------------:{}".format(e-s))
    ```

    数据集按照 https://github.com/Oneflow-Inc/libai/tree/main/projects/T5 下的第三点准备好

    ```bash
    NODE=2 NODE_RANK=0 ADDR=11.11.1.25 bash tools/train.sh tools/train_net.py projects/T5/configs/mt5_pretrain.py 4
    NODE=2 NODE_RANK=1 ADDR=11.11.1.25 bash tools/train.sh tools/train_net.py projects/T5/configs/mt5_pretrain.py 4
    ```