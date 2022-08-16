## 复现Megatron-Deepspeed的流程
### 准备环境

首先准备多机测试环境 参考

[OneFlow性能测试前的标准流程](https://github.com/Oneflow-Inc/OneTeam/issues/478)

[Megatron-LM多机训练](https://github.com/Oneflow-Inc/OneTeam/issues/328#issuecomment-820375669)

```bash
## 参考 https://github.com/bigscience-workshop/Megatron-DeepSpeed

# 克隆仓库
git clone https://github.com/bigscience-workshop/Megatron-DeepSpeed.git
cd Megatron-DeepSpeed

# 安装apex
git clone https://github.com/NVIDIA/apex
cd apex
python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ # 多机都要安装

# 安装deepspeed
cd ..
git clone https://github.com/microsoft/deepspeed
cd deepspeed
rm -rf build
pip install -e . --global-option="build_ext" --global-option="-j8" --no-cache -v --disable-pip-version-check # 多机都要安装
cd ..

pip install transformers
```

### 准备数据集
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
将本仓库gpt_script路径下的 args_deepspeed_gpt.sh 和 args_deepspeed_gpt.sh 拷贝到 Megatron-DeepSpeed 仓库下

以类脑vs008、vs009为例
```bash
bash args_deepspeed_gpt.sh 2 8 0 "10.10.0.8" 2 4 true true 32 512
bash args_deepspeed_gpt.sh 2 8 1 "10.10.0.8" 2 4 true true 32 1024
```

## 复现LiBai+Zero的流程
### 环境准备
```bash
git clone https://github.com/Oneflow-Inc/libai.git
cd libai
git checkout 9fc504c457da4fd1e92d854c60b7271c89a55222
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
将 https://github.com/Oneflow-Inc/OneAutoTest/blob/main/libai/args_libai_gpt2.sh 拷贝至libai仓库的tools路径下，CMD增加zero相关的两行
将 https://github.com/Oneflow-Inc/OneAutoTest/blob/main/libai/gpt2_nl24_nah16_hs1024.py 拷贝至libai仓库的configs路径下，并修改数据集路径
若想对比 LiBai的pipeline_stage_id优化 开启和关闭后的性能，则在 configs/gpt2_nl24_nah16_hs1024.py 中添加和注释掉如下行即可

`train.dist.custom_pipeline_stage_id =  [0] * 6 + [1] * 6 + [2] * 6 + [3] * 6`
```bash
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 2 8 0 "10.10.0.8" 2 4 true true 32 512
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 2 8 1 "10.10.0.8" 2 4 true true 60 1920
```