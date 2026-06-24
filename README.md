# DinoLink

**A Token-Centric Representation Compression Framework for Bandwidth-Constrained Collaborative V2X Perception**

<p align="center">
  <img alt="Conference" src="https://img.shields.io/badge/IROS-2026-blue.svg">
  <img alt="Python" src="https://img.shields.io/badge/python-3.8%2B-brightgreen.svg">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C.svg?logo=pytorch&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey.svg">
</p>

> Tianle Zhu\*, Haohua Que\*, Handong Yao†, Hongyi Xu, Zhipeng Bao — *Accepted to **IROS 2026***
> <sub>\* Equal contribution &nbsp;·&nbsp; † Corresponding author</sub>

---

## Overview

V2X collaborative perception is bottlenecked by bandwidth: human-centric codecs (JPEG/H.264) destroy machine-vision semantics, while raw Float32 feature maps are *larger* than the original images.

**DinoLink** replaces pixel streaming with **discrete semantic communication** via a **dual-sparsity funnel**:

1. **Spatial sparsity** — a *Saliency-Aware Top-K Selector* prunes redundant background tokens.
2. **Bit-level sparsity** — a *Residual Vector Quantization (RVQ)* module collapses surviving features into compact codebook indices.

Transmitting only integer indices + positional priors yields a **139× bitrate reduction** (~1.9 KB/image, 0.021 BPP) at a competitive **32.8% mAP** on nuScenes, with up to **34.5× lower end-to-end latency** in narrow-band links (LoRa, 2G).

**Pipeline:** edge vehicle `(frozen DINOv2 → Top-K selection → RVQ indices)` → V2X link → cloud `(de-quantize → token decoder → DETR backend)`. The DINOv2 encoder stays frozen; the projector, RVQ codebooks, token decoder, and (optionally) the downstream head are trained.

---

## Key Results

On **nuScenes** (5,000 surround-view images projected to 2D COCO boxes, 10 classes, seed 42), DETR as downstream detector:

| Method | BPP ↓ | mAP ↑ | mAP₅₀ ↑ |
|---|---|---|---|
| JPEG (Q100) | 2.641 | 38.8% | 69.2% |
| WebP (Q80) | 0.452 | 38.1% | 69.0% |
| DINO (no comp.) | 2.920 | 38.8% | 69.2% |
| **DinoLink (RVQ)** | **0.021** | 32.8% | 63.0% |

- **Top-K ratio:** 90% is Pareto-optimal (0.019 BPP, 32.4% mAP).
- **RVQ codebook:** size 768 is the sweet spot (61.2% utilization, 32.8% mAP).
- **Latency:** 349.6 s → 10.1 s on LoRa (34.5×); near real-time (~0.07 s) on 5G/WiFi.
- A real-world LAN deployment matches WebP/JPEG accuracy at **500–1000× lower bandwidth**.

---

## Installation

```bash
git clone <your-repo-url> dinolink_project
cd dinolink_project
pip install -r requirements.txt
```

Requires `torch`, `torchvision`, `transformers>=4.30.0`, `pyyaml`, `matplotlib`, `numpy`, `Pillow`. The DINOv2 backbone is downloaded automatically via Hugging Face on first use.

---

## Repository Structure

```
dinolink_project/
├── main.py                       # DinoLink entry point (token compression train/test)
├── configs/config.yaml           # Model / quantizer / decoder / data / run config
├── models/                       # DINOv2 extractor, Top-K selector, projector, RVQ, decoder
├── losses/losses.py              # L2 + Logit-Laplace + commitment losses
├── utils/                        # Dataset loaders & visualizers
├── tools/                        # Latency sim, COCO export, Pareto/convergence plots
├── third_party/detr/             # DETR backend integrated with DinoLink tokens
└── logs/                         # Per-run metrics, figures, checkpoints
```

---

## Usage

**1. Train the token compressor** (edit `configs/config.yaml` first — DINOv2 variant, Top-K, RVQ codebook, dataset root):

```bash
python main.py --config configs/config.yaml --mode train --run_name my_run
```

Checkpoints → `logs/{run_name}/checkpoints/`, metrics → `logs/{run_name}/metrics.txt`.

**2. Evaluate a checkpoint:**

```bash
python main.py --config configs/config.yaml --mode test \
  --ckpt logs/my_run/checkpoints/ckpt_epoch30.pt
```

Reports token reconstruction (`token_cos_sim`) and RVQ effectiveness (`vq_utilization`, `vq_perplexity`, `vq_quant_error`).

**3. Downstream detection with the DETR backend** (consumes DinoLink tokens as queries):

```bash
cd third_party/detr
python main.py --dataset_file coco --coco_path /path/to/nuscenes_coco \
  --use_dinolink_tokens \
  --dinolink_cfg ../../configs/config.yaml \
  --dinolink_ckpt ../../logs/my_run/checkpoints/ckpt_epoch30.pt \
  --freeze_dinolink --output_dir outputs/dinolink_detr
```

Add `--eval` to evaluate, or `--no_freeze_dinolink` to fine-tune DinoLink jointly with DETR.

**4. Reproduce paper analyses:** `tools/export_2d_to_coco.py` (build benchmark), `tools/sim_latency.py` (Fig. 5), `tools/plot_token_pareto.py` (Fig. 4).

---

## Citation

```bibtex
@inproceedings{zhu2026dinolink,
  title     = {DinoLink: A Token-Centric Representation Compression Framework
               for Bandwidth-Constrained Collaborative V2X Perception},
  author    = {Zhu, Tianle and Que, Haohua and Yao, Handong and Xu, Hongyi and Bao, Zhipeng},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026}
}
```

## Acknowledgements & License

Built on [DINOv2](https://github.com/facebookresearch/dinov2) and [DETR](https://github.com/facebookresearch/detr) (vendored under `third_party/detr`). Released under the MIT License; vendored DETR code retains its original Apache-2.0 license.
