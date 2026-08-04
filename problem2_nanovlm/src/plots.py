import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from prompts import LABELS

CHANCE = 1.0 / len(LABELS)


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def confusion(res, path):
    cm = np.array(res["confusion_matrix"], float)
    cmn = cm / cm.sum(1, keepdims=True).clip(1e-9)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if cmn[i, j] > .5 else "black")
    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("predicted"); ax.set_ylabel("gold")
    ax.set_title(f"acc={res['accuracy']:.3f}  kappa={res['cohens_kappa']:.3f}")
    fig.colorbar(im, fraction=.046)
    _save(fig, path)


def reliability(res, path):
    bins = [b for b in res["reliability_bins"] if b["n"]]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(4.4, 5),
                                  gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.plot([b["conf"] for b in bins], [b["acc"] for b in bins], "o-",
            color="#d1495b", label="model")
    ax.axhline(CHANCE, color="gray", ls=":", lw=.8, label="chance")
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1); ax.legend(fontsize=7)
    ax.set_title(f"ECE={res['ece']:.3f}   mean confidence={res['mean_confidence']:.3f}")
    ax2.bar([b["lo"] for b in bins], [b["n"] for b in bins], width=.06,
            align="edge", color="#4a6fa5")
    ax2.set_xlabel("confidence"); ax2.set_ylabel("count"); ax2.set_xlim(0, 1)
    _save(fig, path)


def per_template(res, path, baseline=None):
    per = res.get("per_template") or {}
    if not per:
        return
    ks = sorted(per, key=lambda k: -per[k]["accuracy"])
    x = np.arange(len(ks)); w = .38 if baseline else .7
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(ks)), 4))
    if baseline:
        bp = baseline.get("per_template", {})
        ax.bar(x - w / 2, [bp.get(k, {}).get("accuracy", np.nan) for k in ks], w,
               label="baseline", color="#9aa5b1")
        ax.bar(x + w / 2, [per[k]["accuracy"] for k in ks], w,
               label="fine-tuned", color="#2a9d8f")
        ax.legend(fontsize=8)
    else:
        ax.bar(x, [per[k]["accuracy"] for k in ks], w, color="#2a9d8f")
    ax.axhline(CHANCE, color="k", ls="--", lw=.8)
    ax.set_xticks(x, [f"{k[:30]}\n(n={per[k]['n']})" for k in ks],
                  rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1)
    _save(fig, path)


def confidence_hist(rows, path):
    p = np.array([r["probs"] for r in rows])
    g = np.array([r["gold"] for r in rows])
    ok = p.argmax(1) == g
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.hist(p.max(1)[ok], bins=20, alpha=.65, label="correct", color="#2a9d8f")
    ax.hist(p.max(1)[~ok], bins=20, alpha=.65, label="wrong", color="#d1495b")
    ax.set_xlabel("confidence"); ax.set_ylabel("count")
    ax.legend(fontsize=8)
    _save(fig, path)


def training_curves(csv_path, path):
    import csv as _csv
    rows = [r for r in _csv.DictReader(open(csv_path)) if r["step"].isdigit()]
    if not rows:
        return
    s = list(range(1, len(rows) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
    axes[0].plot(s, [float(r["train_loss"]) for r in rows], "o-", label="train", ms=3)
    axes[0].plot(s, [float(r["val_loss"]) for r in rows], "s--", label="val", ms=3)
    axes[0].set_title("loss"); axes[0].legend(fontsize=7)
    axes[1].plot(s, [float(r["train_acc"]) for r in rows], "o-", label="train", ms=3)
    axes[1].plot(s, [float(r["val_acc"]) for r in rows], "s--", label="val", ms=3)
    axes[1].axhline(CHANCE, color="k", ls=":", lw=.8)
    axes[1].set_title("accuracy"); axes[1].legend(fontsize=7)
    axes[2].plot(s, [float(r["val_ppl"]) for r in rows], color="#d1495b")
    axes[2].set_title("validation perplexity")
    for ax in axes:
        ax.set_xlabel("checkpoint")
    _save(fig, path)


def baseline_vs_ft(base, ft, path):
    keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1",
            "cohens_kappa", "ece"]
    keys = [k for k in keys if k in base or k in ft]
    x = np.arange(len(keys)); w = .38
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(x - w / 2, [base.get(k, 0) for k in keys], w, label="baseline",
           color="#9aa5b1")
    ax.bar(x + w / 2, [ft.get(k, 0) for k in keys], w, label="fine-tuned",
           color="#2a9d8f")
    for i, k in enumerate(keys):
        ax.text(i - w / 2, base.get(k, 0) + .01, f"{base.get(k, 0):.3f}",
                ha="center", fontsize=7)
        ax.text(i + w / 2, ft.get(k, 0) + .01, f"{ft.get(k, 0):.3f}",
                ha="center", fontsize=7)
    ax.set_xticks(x, [k.replace("_", " ") for k in keys], rotation=15, fontsize=8)
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8)
    _save(fig, path)


def eval_plots(res, rows, out_dir):
    confusion(res, os.path.join(out_dir, "confusion_matrix.png"))
    reliability(res, os.path.join(out_dir, "reliability.png"))
    per_template(res, os.path.join(out_dir, "per_template.png"))
    confidence_hist(rows, os.path.join(out_dir, "confidence_hist.png"))