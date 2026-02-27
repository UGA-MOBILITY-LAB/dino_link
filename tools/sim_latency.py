#!/usr/bin/env python3
"""
Simulate end-to-end latency for three deployment modes:
  1) All_Local
  2) All_Server
  3) Partition (edge -> send VQ payload -> server)

This script reads DinoLink config to compute VQ payload size, then evaluates
latency under multiple communication links (2G/3G/4G/5G/WiFi) and draws a
bar chart similar to the provided example figure.
"""

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple


DINOV2_PATCH = 14

# Typical link presets: (uplink_mbps, downlink_mbps, rtt_ms)
LINK_PRESETS = {
    "2G": (0.1, 0.2, 600.0),
    "3G": (1.0, 5.0, 150.0),
    "4G": (10.0, 20.0, 50.0),
    "5G": (50.0, 200.0, 20.0),
    "WiFi": (30.0, 100.0, 15.0),
}

# Image upload codec presets, expressed as ratio to raw bytes (H*W*C).
# These are practical defaults for latency simulation, not strict guarantees.
CODEC_RATIO_PRESETS = {
    "RAW": 1.00,
    "JPEG_Q100": 0.20,
    "JPEG_Q90": 0.10,
    "JPEG_Q80": 0.06,
    "WEBP_Q90": 0.07,
    "WEBP_Q80": 0.05,
    "AVIF_Q50": 0.04,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Latency simulation for DinoLink edge/server split")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint (existence check; kept fixed for same-run comparison)",
    )
    parser.add_argument(
        "--output_png",
        type=str,
        default="outputs/latency_sim/latency_bar.png",
        help="Output bar chart path",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="outputs/latency_sim/latency_table.csv",
        help="Output table path",
    )
    parser.add_argument(
        "--links",
        type=str,
        default="2G,3G,4G,5G,WiFi",
        help="Comma-separated link names from {2G,3G,4G,5G,WiFi}",
    )
    parser.add_argument(
        "--all_local_compute_s",
        type=float,
        default=7.113,
        help="Measured local full-pipeline compute latency in seconds",
    )
    parser.add_argument(
        "--all_server_compute_s",
        type=float,
        default=0.060,
        help="Measured server full-pipeline compute latency in seconds",
    )
    parser.add_argument(
        "--edge_partition_compute_s",
        type=float,
        default=0.600,
        help="Measured edge-side compute latency before transmission in partition mode",
    )
    parser.add_argument(
        "--server_partition_compute_s",
        type=float,
        default=0.080,
        help="Measured server-side compute latency after receiving partition payload",
    )
    parser.add_argument(
        "--result_bytes",
        type=float,
        default=1200.0,
        help="Result payload size in bytes (e.g. boxes/classes/scores)",
    )
    parser.add_argument(
        "--image_bytes",
        type=float,
        default=0.0,
        help=(
            "If >0, directly use this uploaded image size in bytes for all-server mode. "
            "Highest priority override over codec/raw estimation."
        ),
    )
    parser.add_argument(
        "--image_codec",
        type=str,
        default="RAW",
        choices=sorted(CODEC_RATIO_PRESETS.keys()),
        help=(
            "Codec preset for all-server upload size estimation. "
            "Used only when --image_bytes <= 0."
        ),
    )
    parser.add_argument(
        "--codec_scale",
        type=float,
        default=1.0,
        help=(
            "Scale factor on top of codec preset ratio (e.g. 1.2 means 20%% larger files). "
            "Used only when --image_bytes <= 0."
        ),
    )
    parser.add_argument("--image_h", type=int, default=900, help="Raw image height if image_bytes=0")
    parser.add_argument("--image_w", type=int, default=1600, help="Raw image width if image_bytes=0")
    parser.add_argument("--image_c", type=int, default=3, help="Raw image channels if image_bytes=0")
    parser.add_argument(
        "--use_dinov2_size_for_image",
        action="store_true",
        help="Use model.dinov2_image_size from config as image H/W for all-server upload bytes",
    )
    return parser.parse_args()


def load_cfg(path: Path) -> Dict:
    try:
        import yaml
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "pyyaml is required to read config yaml. "
            "Install it in your env (e.g. `pip install pyyaml`)."
        ) from e
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_num_patches(cfg: Dict, image_h: int, image_w: int) -> int:
    model_cfg = cfg.get("model", {})
    dino_size = model_cfg.get("dinov2_image_size", None)
    if dino_size is not None:
        nph = int(dino_size) // DINOV2_PATCH
        npw = int(dino_size) // DINOV2_PATCH
    else:
        nph = int(image_h) // DINOV2_PATCH
        npw = int(image_w) // DINOV2_PATCH
    return max(nph * npw, 1)


def calc_partition_payload_bytes(cfg: Dict, num_patches: int) -> Tuple[float, Dict[str, float]]:
    model_cfg = cfg.get("model", {})
    quant_cfg = cfg.get("quantizer", {})

    top_k = int(model_cfg.get("top_k", 0))
    codebook_size = int(quant_cfg.get("codebook_size", 2))
    quant_type = str(quant_cfg.get("type", "vq")).lower()
    if quant_type == "rvq":
        num_quantizers = int(quant_cfg.get("num_quantizers", 2))
    elif quant_type in {"vq"}:
        num_quantizers = 1
    else:
        # For non-VQ types, keep formula stable but indicates no quantization.
        num_quantizers = 1

    bits_code = int(math.ceil(math.log2(max(codebook_size, 2)))) * num_quantizers
    bits_pos = int(math.ceil(math.log2(max(num_patches, 2))))
    bits_per_token = bits_code + bits_pos
    payload_bytes = float(top_k * bits_per_token) / 8.0

    detail = {
        "top_k": float(top_k),
        "num_patches": float(num_patches),
        "codebook_size": float(codebook_size),
        "num_quantizers": float(num_quantizers),
        "bits_code": float(bits_code),
        "bits_pos": float(bits_pos),
        "bits_per_token": float(bits_per_token),
    }
    return payload_bytes, detail


def transmission_time_s(payload_bytes: float, mbps: float) -> float:
    bps = mbps * 1e6
    return (payload_bytes * 8.0) / bps


def simulate(
    links: List[str],
    image_bytes: float,
    partition_payload_bytes: float,
    result_bytes: float,
    all_local_compute_s: float,
    all_server_compute_s: float,
    edge_partition_compute_s: float,
    server_partition_compute_s: float,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for link in links:
        ul, dl, rtt_ms = LINK_PRESETS[link]
        rtt_s = rtt_ms / 1000.0

        upload_image_s = transmission_time_s(image_bytes, ul)
        upload_partition_s = transmission_time_s(partition_payload_bytes, ul)
        download_result_s = transmission_time_s(result_bytes, dl)

        all_local_s = all_local_compute_s
        all_server_s = upload_image_s + rtt_s + all_server_compute_s + download_result_s
        partition_s = (
            edge_partition_compute_s
            + upload_partition_s
            + rtt_s
            + server_partition_compute_s
            + download_result_s
        )

        rows.append(
            {
                "link": link,
                "all_local_s": all_local_s,
                "all_server_s": all_server_s,
                "partition_s": partition_s,
                "upload_image_s": upload_image_s,
                "upload_partition_s": upload_partition_s,
                "download_result_s": download_result_s,
                "rtt_s": rtt_s,
            }
        )
    return rows


def save_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "link",
        "all_local_s",
        "all_server_s",
        "partition_s",
        "upload_image_s",
        "upload_partition_s",
        "download_result_s",
        "rtt_s",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def draw_bar(path: Path, rows: List[Dict[str, float]], title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. "
            "Install it in your env (e.g. `pip install matplotlib`) or use the CSV output only."
        ) from e

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [r["link"] for r in rows]
    all_local = [r["all_local_s"] for r in rows]
    all_server = [r["all_server_s"] for r in rows]
    partition = [r["partition_s"] for r in rows]

    x = range(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9.5, 4.8))

    b1 = ax.bar([i - width for i in x], all_local, width=width, label="All_Local")
    b2 = ax.bar(list(x), all_server, width=width, label="All_Server")
    b3 = ax.bar([i + width for i in x], partition, width=width, label="Partition")

    ax.set_ylabel("Latency (s)")
    ax.set_xlabel("Communication link")
    ax.set_title(title)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bars in (b1, b2, b3):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    cfg_path = Path(args.config)
    ckpt_path = Path(args.checkpoint)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cfg = load_cfg(cfg_path)
    model_cfg = cfg.get("model", {})
    if args.use_dinov2_size_for_image and model_cfg.get("dinov2_image_size", None) is not None:
        s = int(model_cfg["dinov2_image_size"])
        image_h, image_w = s, s
    else:
        image_h, image_w = args.image_h, args.image_w

    raw_image_bytes = float(image_h * image_w * args.image_c)
    image_bytes_mode = "raw_estimate"
    codec_ratio = 1.0
    if args.image_bytes > 0:
        image_bytes = float(args.image_bytes)
        image_bytes_mode = "manual_bytes"
    else:
        codec_ratio = float(CODEC_RATIO_PRESETS[args.image_codec]) * float(args.codec_scale)
        image_bytes = raw_image_bytes * codec_ratio
        image_bytes_mode = f"codec:{args.image_codec}"

    num_patches = infer_num_patches(cfg, image_h, image_w)
    partition_payload_bytes, detail = calc_partition_payload_bytes(cfg, num_patches)

    links = [x.strip() for x in args.links.split(",") if x.strip()]
    invalid = [x for x in links if x not in LINK_PRESETS]
    if invalid:
        raise ValueError(f"Unknown links {invalid}. Allowed: {sorted(LINK_PRESETS.keys())}")

    rows = simulate(
        links=links,
        image_bytes=image_bytes,
        partition_payload_bytes=partition_payload_bytes,
        result_bytes=float(args.result_bytes),
        all_local_compute_s=float(args.all_local_compute_s),
        all_server_compute_s=float(args.all_server_compute_s),
        edge_partition_compute_s=float(args.edge_partition_compute_s),
        server_partition_compute_s=float(args.server_partition_compute_s),
    )

    out_png = Path(args.output_png)
    out_csv = Path(args.output_csv)
    save_csv(out_csv, rows)

    title = (
        f"Latency Simulation (cfg={cfg_path.name}, ckpt={ckpt_path.name})\n"
        f"top_k={int(detail['top_k'])}, bits/token={int(detail['bits_per_token'])}, "
        f"partition_payload={partition_payload_bytes:.1f}B, image_bytes={image_bytes:.1f}B ({image_bytes_mode})"
    )
    draw_bar(out_png, rows, title)

    print("=== Simulation Summary ===")
    print(f"config: {cfg_path}")
    print(f"checkpoint: {ckpt_path}")
    print(f"image_hxwxc: {image_h}x{image_w}x{args.image_c}")
    print(f"raw_image_bytes: {raw_image_bytes:.1f}")
    print(f"image_codec: {args.image_codec}")
    print(f"codec_scale: {float(args.codec_scale):.3f}")
    print(f"image_bytes_mode: {image_bytes_mode}")
    if args.image_bytes <= 0:
        print(f"effective_codec_ratio: {codec_ratio:.5f}")
    print(f"image_bytes: {image_bytes:.1f}")
    print(f"num_patches: {int(detail['num_patches'])}")
    print(f"top_k: {int(detail['top_k'])}")
    print(f"bits_code: {int(detail['bits_code'])}")
    print(f"bits_pos: {int(detail['bits_pos'])}")
    print(f"bits_per_token: {int(detail['bits_per_token'])}")
    print(f"partition_payload_bytes: {partition_payload_bytes:.1f}")
    print(f"saved_png: {out_png}")
    print(f"saved_csv: {out_csv}")


if __name__ == "__main__":
    main()
