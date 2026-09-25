# SSP + Consensus

## Files

| 文件 | 用途 |
|---|---|
| `train.py` | 训练、验证、测试和最佳模型选择入口 |
| `model.py` | 模型与类别 Embedding |
| `losses.py` | SSP 损失及 EMA/history 更新 |
| `load_data.py` | 数据划分、候选标签、DataLoader |
| `utils.py` | CLI 参数解析 |
| `evaluate.py` | 检索 mAP |
| `checkpoint_io.py` | 最佳验证 checkpoint 保存 |
| `neighbor_refine.py` | Consensus、KNN 和 memory bank |
| `scripts/` | Linux bash 正式启动脚本 |
| `partial_labels/` | Wiki L5、INRIA L2–5 的已有候选缓存 |
| `results/` | 新实验日志和 checkpoint；交付时只有 `.gitkeep` |
| `.gitignore` | 排除数据、运行产物与 Python 缓存，保留候选缓存 |

## Data

使用已配好 CUDA 的 Python 环境，需要 torch、torchvision、numpy、scipy、h5py。原始数据不随本目录提供，按 `load_data.py` 的实际路径放置（Linux 区分大小写）：

- Wiki：`datasets/wiki.mat`
- INRIA：`datasets/inria-websearch.mat`
- NUS-WIDE：`datasets/nus_wide_deep_doc2vec_data_42941.h5py`
- XMediaNet：`datasets/xmedianet.mat`

代码仍从 `results/partial_labels/` 读取缓存。bash 脚本会把对应的 `partial_labels/*.mat` 复制到该位置，不覆盖已有缓存。直接运行 `train.py` 前，先在项目根目录准备一次：

```bash
mkdir -p results/partial_labels
cp -n partial_labels/*.mat results/partial_labels/
```

## Run

当前正式主线：`--neighbor_mode consensus`；原 SSP：`--neighbor_mode off`。只使用这两个模式，其余可选开关保持默认关闭。

```bash
bash scripts/run_wiki_consensus.sh 123
bash scripts/run_inria_consensus.sh 123
```

Wiki 单次运行 L5，参数沿用当前 Wiki 启动脚本与固定设置：180 epochs、batch_size=2048、output_dim=1024、lr=1e-4、lamda=0.1、ema_decay=0.95。

INRIA 按当前 `run_INRIA-Websearch.sh` 依次运行 L2、L3、L4、L5，lr=1e-4；保留 180 epochs、batch_size=2048、output_dim=1024、lamda=0.1。本次按明确要求，脚本显式传入 ema_decay=0.95；`utils.py` 的 CLI 默认值不改。

两脚本均固定 `independent_train_seed`，Consensus k=10、beta=0.2、margin=0.10、support_threshold=0.20。每次运行创建独立时间戳结果目录并保存 `best_validation.pt`；可用 `PYTHON=/path/to/python bash scripts/run_wiki_consensus.sh 123` 指定解释器。

直接运行原 SSP Wiki L5（先按 Data 一节准备缓存）：

```bash
run_dir="results/ssp/wiki_L5_seed123_$(date +%Y%m%d_%H%M%S_%N)"
mkdir -p results/ssp
mkdir "$run_dir"
python -B -u train.py --dataset wiki --partial_length 5 --seed 123 \
    --MAX_EPOCH 180 --batch_size 2048 --output_dim 1024 \
    --lr 1e-4 --lamda 0.1 --ema_decay 0.95 \
    --neighbor_mode off --independent_train_seed \
    --log_dir "$run_dir" --best_checkpoint "$run_dir/best_validation.pt"
```
