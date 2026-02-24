# DinoLink

DINOv2 token extraction → Top-K / attention-based selection → VQ / EMA-VQ → Decoder.

**详细流程说明**（数据→模型→损失→训练/测试）见 **[PROJECT_FLOW.md](PROJECT_FLOW.md)**。

## Structure

```
dinolink_project/
├── main.py                 # Entry point
├── configs/
│   └── config.yaml         # Model (TopK=100, Codebook=512), data, run
├── models/
│   ├── dinov2_extractor.py # DINOv2 token extractor
│   ├── token_selector.py   # Top-K or attention-based selection
│   ├── quantizer.py        # VQ / EMA-VQ
│   └── decoder.py          # TokenDecoder: z_q -> DINO token
├── utils/
│   ├── image_loader.py    # nuScenes / Waymo / COCO image loader
│   └── visualizer.py      # Attention / selection / token metric matrices
└── logs/
    └── run_xx/            # Per-run: metrics, figs, ckpts
```

## Config

Edit `configs/config.yaml`:

- **model**: `top_k`, `selection`, `dinov2_name`, `dinov2_image_size`  
  - 若 attention map 效果差：可试 `dinov2_name: "facebook/dinov2-with-registers-base"`（需 transformers 较新版本）；热力图已用 98 百分位缩放以增强对比。
- **quantizer**: `type` (vq / rvq), `codebook_size`, `z_dim`
- **decoder**: `type` (token), `hidden_dim`, `use_pos`
- **data**: `dataset` (nuscenes / waymo / coco), `data_root`, `batch_size`
- **run**: `epochs`, `lr`, `token_weight`, `beta`, `entropy_weight`, `log_dir`, `save_every`

### 增加 patch token 个数

- **候选 patch 总数**：由 `model.dinov2_image_size` 决定。224 → 16×16=256 个 patch；448 → 32×32=1024 个 patch（显存占用更大）。
- **实际使用的 token 数**：`min(model.top_k, N)`，其中 `N` 是候选 patch 总数。
- 若要**更多 token**（例如 256 个）：直接将 `model.top_k: 256`。可选同时设 `dinov2_image_size: 448` 以从更多 patch 里做 top-k。

## Run

### Training

From project root:

```bash
cd dinolink_project
python main.py --config configs/config.yaml --mode train
```

Optional: `--run_name my_run` to set log subdir (default: `run_YYYYmmdd_HHMMSS`).  
Checkpoints are saved under `logs/{run_name}/checkpoints/ckpt_epoch{N}.pt`.

### Test (evaluate with checkpoint)

**方式一：指定 checkpoint 路径**

```bash
python main.py --config configs/config.yaml --mode test --ckpt logs/run_xxx/checkpoints/ckpt_epoch10.pt
```

**方式二：用 run_name + epoch 号（不写则用最新）**

```bash
# 用 run_xxx 下 epoch 10 的 ckpt
python main.py --config configs/config.yaml --mode test --run_name run_xxx --ckpt_epoch 10

# 用 run_xxx 下按时间最新的 ckpt
python main.py --config configs/config.yaml --mode test --run_name run_xxx
```

Test 会：加载权重 -> 全模型 `eval()` -> 跑一遍数据算 token/VQ 指标 -> 写入 `logs/{run_name}/test_metrics.txt` 并保存 attention、selection、token metric matrices。

## 评估方式 (Evaluation)

运行（训练或测试）时，每个 batch 会走一遍：**DINOv2 提 token → Top-K 选择 → 投影 → VQ 量化 → 解码重建**，并计算下面这些指标（在 `run_step` 里算，训练时每 epoch 平均后打印并写入 `metrics.txt`，测试时全量数据平均后打印并写入 `test_metrics.txt`）。

### 1. Token 重建与损失

| 指标 | 含义 | 计算方式 |
|------|------|----------|
| **token_loss** | token 重建误差 | MSE(decoded_tokens, selected_tokens) |
| **token_mse_sel** | 选中 token 平均 MSE | 对每个选中 token 的特征维做均值再 batch 平均 |
| **token_cos_sim** | token 语义相似度 | cosine(decoded_tokens, selected_tokens) 的平均值（越高越好） |
| **commit_loss** | VQ 约束损失 | `vq` / `rvq` 对应的 commitment 项 |
| **loss** | 总损失 | `token_weight*token_loss + commit_loss + entropy_weight*diversity_loss` |

### 2. VQ 压缩有效性（见下方说明）

| 指标 | 含义 | 计算方式 |
|------|------|----------|
| **vq_utilization** | Codebook 利用率 | 本 batch 被用到的 code 数 / codebook_size，∈[0,1] |
| **vq_perplexity** | 有效码数（熵的 exp） | 基于各 code 使用频率的熵，再 exp；越高表示码本用得越均匀 |
| **vq_quant_error** | 量化误差 | MSE(z_enc, z_q)，量化前后 latent 的误差 |
| **vq_active_codes** | 激活的 code 数 | 本 batch 里出现过的不同 code 数量 |

### 3. 输出与可视化（Evaluation Matrix）

- **训练**：每个 epoch 平均指标写入 `logs/{run_name}/metrics.txt`；每 `save_every` 个 epoch 保存：
  - `attention_epoch*.png`
  - `selection_epoch*.png`
  - `token_matrices_epoch*.png`（Token MSE / VQ Error / Selection Mask）
  - checkpoint
- **测试**：全量数据平均后写入 `logs/{run_name}/test_metrics.txt`，并保存：
  - `test_attention.png`
  - `test_selection.png`
  - `test_token_matrices.png`

**总结**：评估完全基于「token 重建质量 + VQ 损失 + 码本利用情况」，没有引入下游任务或 FID。


### 本地 / 相对路径

Place images under `data_root` (or override with `--data_root`):

- **nuScenes**: `data_root/samples/CAM_FRONT/` or `data_root/images/`
- **Waymo**: `data_root/images/` or `data_root/camera_image/`
- **COCO**: `data_root/train2017/` or `data_root/val2017/`

If the path does not exist, the script falls back to dummy tensors for a quick sanity run.

## VQ 压缩有效性 (Compression effectiveness)

训练时每个 epoch 会计算并打印/写入日志的 VQ 指标：

- **vq_utilization**: codebook 利用率 = 被用到的 code 数量 / codebook_size，越接近 1 说明码本利用越充分。
- **vq_perplexity**: exp(entropy)，有效使用的 code 数量；越高表示码本使用越均匀、信息量越大。
- **vq_quant_error**: 量化误差 MSE(z_enc, z_q)，越小表示 VQ 重建越准。
- **vq_active_codes**: 当前 batch 中被激活的 code 数量。

这些指标会写入 `logs/run_xx/metrics.txt`，便于评估 VQ 压缩是否有效（利用率与 perplexity 不宜长期接近 0，quant_error 应随训练下降）。
