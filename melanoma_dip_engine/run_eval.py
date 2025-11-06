from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .metrics import evaluate_directory, save_csv, save_json


def main(gt_dir: str, pred_dir: str, out_dir: str) -> None:
    gt = Path(gt_dir)
    pr = Path(pred_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows, avg = evaluate_directory(gt, pr)
    save_csv(rows, out / "eval_report.csv")
    save_json({"average": avg, "count": len(rows)}, out / "eval_report.json")

    # Quick bar plot
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.bar(["Dice", "IoU", "Prec", "Rec"], [avg["dice"], avg["iou"], avg["precision"], avg["recall"]])
    ax.set_ylim(0, 1)
    ax.set_title("Validation Metrics")
    fig.tight_layout()
    fig.savefig(out / "eval_report.png", dpi=150)
    plt.close(fig)

    print(f"Wrote reports to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", default="melanoma_dip_engine/data/val/masks")
    parser.add_argument("--pred", default="outputs/preds")
    parser.add_argument("--out", default="models")
    args = parser.parse_args()
    main(args.gt, args.pred, args.out)



