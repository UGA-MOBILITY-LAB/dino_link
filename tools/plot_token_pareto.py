#!/usr/bin/env python3
"""
Plot BPP-AP trade-off and Pareto frontier from DETR JSONL logs.

Example:
  python tools/plot_token_pareto.py \
    --point 50:/home/tianle/dinolink_project/outputs/nuSenes2d_dinolink_w_VQ_top_50/nuSenes2d_dinolink_w_VQ_top_50_log.txt \
    --point 70:/home/tianle/dinolink_project/outputs/nuSenes2d_dinolink_w_VQ_top_70/log.txt \
    --point 90:/home/tianle/dinolink_project/outputs/nuSenes2d_dinolink_w_VQ_top_90/log.txt \
    --point 100:/home/tianle/dinolink_project/outputs/nuSenes2d_dinolink_w_VQ/nuSenes2d_dinolink_w_VQ_log.txt \
    --output /home/tianle/dinolink_project/outputs/pareto_bpp_ap.png
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Plot BPP-AP Pareto frontier")
    p.add_argument(
        "--point",
        action="append",
        default=[],
        help="One experiment point in format label:log_path (e.g. 70:/path/log.txt). Repeatable.",
    )
    p.add_argument(
        "--output",
        type=str,
        default="outputs/pareto_bpp_ap.png",
        help="Output PNG path.",
    )
    p.add_argument(
        "--title",
        type=str,
        default="Pareto Frontier: BPP vs AP",
        help="Plot title.",
    )
    return p.parse_args()


def _to_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def extract_best_metrics(log_path: Path) -> Dict[str, Optional[float]]:
    best_ap = None
    best_line = None
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except json.JSONDecodeError:
                continue
            coco = rec.get("test_coco_eval_bbox", None)
            if not isinstance(coco, list) or len(coco) == 0:
                continue
            ap = _to_float(coco[0])
            if ap is None:
                continue
            if best_ap is None or ap > best_ap:
                best_ap = ap
                best_line = rec

    if best_line is None:
        raise RuntimeError(f"No valid test_coco_eval_bbox records found in: {log_path}")

    bpp = None
    for k in ("test_input_bpp", "input_bpp", "train_input_bpp"):
        if k in best_line:
            bpp = _to_float(best_line.get(k))
            if bpp is not None:
                break

    return {
        "ap": best_ap,
        "bpp": bpp,
        "epoch": _to_float(best_line.get("epoch")),
    }


def parse_points(items: List[str]) -> List[Tuple[str, Path]]:
    points: List[Tuple[str, Path]] = []
    for it in items:
        if ":" not in it:
            raise ValueError(f"Invalid --point format: {it}. Expected label:log_path")
        label, path_s = it.split(":", 1)
        path = Path(path_s).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Log not found: {path}")
        points.append((label, path))
    return points


def build_pareto_frontier(rows: List[Dict[str, Optional[float]]]) -> List[Dict[str, Optional[float]]]:
    """Return non-dominated points for objective: minimize bpp, maximize ap."""
    valid = [r for r in rows if r["bpp"] is not None]
    valid.sort(key=lambda x: (float(x["bpp"]), -float(x["ap"])))
    frontier: List[Dict[str, Optional[float]]] = []
    best_ap_so_far = -1e9
    for r in valid:
        ap = float(r["ap"])
        if ap > best_ap_so_far:
            frontier.append(r)
            best_ap_so_far = ap
    return frontier


def main() -> None:
    args = parse_args()
    if not args.point:
        raise ValueError("Please provide at least one --point token_pct:log_path")

    points = parse_points(args.point)
    rows = []
    for label, log_path in points:
        m = extract_best_metrics(log_path)
        token_pct = _to_float(label)
        rows.append(
            {
                "label": label,
                "token_pct": token_pct,
                "log_path": str(log_path),
                "ap": float(m["ap"]),
                "bpp": m["bpp"],
                "epoch": m["epoch"],
            }
        )
    rows.sort(key=lambda x: (x["bpp"] is None, x["bpp"] if x["bpp"] is not None else 1e18))

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "matplotlib is required to draw the Pareto plot. Install via `pip install matplotlib`."
        ) from e

    xs = [float(r["bpp"]) for r in rows if r["bpp"] is not None]
    ys = [float(r["ap"]) for r in rows if r["bpp"] is not None]
    if not xs:
        raise RuntimeError("No point has valid BPP. Check logs for test_input_bpp/train_input_bpp.")

    frontier = build_pareto_frontier(rows)
    fx = [float(r["bpp"]) for r in frontier]
    fy = [float(r["ap"]) for r in frontier]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.scatter(xs, ys, marker="o", s=36, alpha=0.75, label="All points")
    ax.plot(fx, fy, color="tab:red", marker="o", linewidth=2.0, label="Pareto frontier")
    ax.set_xlabel("Input BPP")
    ax.set_ylabel("Best AP@[0.50:0.95]")
    ax.set_title(args.title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    for r in rows:
        if r["bpp"] is None:
            continue
        label = f"{r['label']}\nAP={r['ap']:.3f}"
        ax.annotate(
            label,
            xy=(r["bpp"], r["ap"]),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)

    print("Saved:", out)
    print("\nPoints:")
    for r in rows:
        flag = " [Pareto]" if r in frontier else ""
        print(
            f"  label={r['label']} | AP={r['ap']:.4f} | "
            f"BPP={r['bpp'] if r['bpp'] is not None else 'N/A'} | epoch={r['epoch']}{flag}"
        )


if __name__ == "__main__":
    main()

