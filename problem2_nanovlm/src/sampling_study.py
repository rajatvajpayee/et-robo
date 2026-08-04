import argparse
import collections
import csv
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from prompts import LABELS
from sampling import generate_one

import setproctitle
setproctitle.setproctitle("sampling_study.py")

STRATEGIES = [
    {"name": "greedy", "mode": "greedy"},
    {"name": "T=0.5", "mode": "temperature", "temperature": 0.5},
    {"name": "T=1.0","mode": "temperature", "temperature": 1.0},
    {"name": "T=1.5","mode": "temperature", "temperature": 1.5},
    {"name": "top-k=1","mode": "top_k", "k": 1},
    {"name": "top-k=5",  "mode": "top_k", "k": 5},
    {"name": "top-k=50", "mode": "top_k", "k": 50},
    {"name": "top-p=0.5", "mode": "top_p", "p": 0.5},
    {"name": "top-p=0.9", "mode": "top_p", "p": 0.9},
    {"name": "top-p=0.99","mode": "top_p", "p": 0.99},
    {"name": "min-p=0.05","mode": "min_p", "p": 0.05},
    {"name": "min-p=0.1", "mode": "min_p", "p": 0.1},
    {"name": "min-p=0.1 T=2", "mode": "min_p", "p": 0.1, "temperature": 2.0},
]


def decode_label(text, n_choices):
    """First label character in the output, or -1 if none / out of range."""
    for ch in text.strip():
        if ch in LABELS:
            i = LABELS.index(ch)
            return i if i < n_choices else -1
    return -1


def flatten_examples(loader, device, max_batches=None):
    """One entry per test example, prompt sliced at the answer position."""
    examples = []
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        for g in batch:
            images = g["images"].to(device)
            for j in range(g["input_ids"].size(0)):
                end = int(g["answer_pos"][j])          # generate from here on
                examples.append({
                    "id": g["ids"][j],
                    "input_ids": g["input_ids"][j:j + 1, :end].to(device),
                    "attention_mask": g["attention_mask"][j:j + 1, :end].to(device),
                    "images": images[j:j + 1].flatten(0, 1),
                    "gold": int(g["answer_idx"][j]),
                    "n_choices": int(g["n_choices"][j]),
                })
    return examples


def run_strategy(model, examples, strategy, seeds, max_new_tokens, device):
    kw = {k: v for k, v in strategy.items()
          if k in ("mode", "temperature", "k", "p")}
    eos = model.tokenizer.eos_token_id

    outputs = collections.defaultdict(list)   # example id -> [decoded text, ...]
    entropies, supports = [], []

    for seed in seeds:
        gen = torch.Generator(device=device).manual_seed(seed)
        for ex in tqdm(examples, leave=False, desc=strategy["name"]):
            ids, entropy, support = generate_one(
                model, ex["input_ids"], ex["images"], ex["attention_mask"],
                max_new_tokens=max_new_tokens, eos_id=eos, generator=gen, **kw)
            entropies.append(entropy)
            supports.append(support)
            outputs[ex["id"]].append(
                model.tokenizer.decode(ids[0].tolist(), skip_special_tokens=True))

    by_id = {ex["id"]: ex for ex in examples}
    accuracy, valid, consistency, distinct = [], [], [], []

    for ex_id, texts in outputs.items():
        ex = by_id[ex_id]
        preds = [decode_label(t, ex["n_choices"]) for t in texts]
        accuracy.append(np.mean([p == ex["gold"] for p in preds]))
        valid.append(np.mean([p >= 0 for p in preds]))
        modal = collections.Counter(preds).most_common(1)[0][1]
        consistency.append(modal / len(preds))
        distinct.append(len(set(texts)))

    return {
        "strategy": strategy["name"],
        "mode": strategy["mode"],
        "temperature": strategy.get("temperature", 1.0),
        "k": strategy.get("k", ""),
        "p": strategy.get("p", ""),
        "accuracy": float(np.mean(accuracy)),
        "valid_label_rate": float(np.mean(valid)),
        "self_consistency": float(np.mean(consistency)),
        "distinct_outputs": float(np.mean(distinct)),
        "mean_support": float(np.mean(supports)),
        "mean_entropy": float(np.mean(entropies)),
    }


def save(rows, out_dir, n_examples, seeds):
    os.makedirs(out_dir, exist_ok=True)
    keys = list(rows[0])

    with open(os.path.join(out_dir, "sampling_table.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(rows)

    md = [f"n = {n_examples} examples, {len(seeds)} seeds", "",
          "| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in rows:
        md.append("| " + " | ".join(
            f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k])
            for k in keys) + " |")
    with open(os.path.join(out_dir, "sampling_table.md"), "w") as f:
        f.write("\n".join(md))

    with open(os.path.join(out_dir, "sampling.json"), "w") as f:
        json.dump({"n_examples": n_examples, "seeds": list(seeds), "rows": rows},
                  f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.bar(x - .2, [r["accuracy"] for r in rows], .4,
           label="accuracy", color="#2a9d8f")
    ax.bar(x + .2, [r["self_consistency"] for r in rows], .4,
           label="self-consistency", color="#9aa5b1")
    ax.set_xticks(x, [r["strategy"] for r in rows],
                  rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("rate")
    ax.legend(fontsize=7, loc="lower left")

    ax2 = ax.twinx()
    ax2.plot(x, [r["mean_support"] for r in rows], "o--", color="#d1495b", ms=4)
    ax2.set_yscale("log")
    ax2.set_ylabel("mean surviving tokens")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sampling_effects.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    from dataset import Robo2VLMDataModule
    from model import NanoVLM

    ap = argparse.ArgumentParser()
    ap.add_argument("--train-path", default="/home/rajat/scratch/et-robo/data/Robo2VLM-1_local/train")
    ap.add_argument("--test-path", default="/home/rajat/scratch/et-robo/data/Robo2VLM-1_local/test")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default="logs/sampling")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--max-batches", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NanoVLM(checkpoint=args.checkpoint, device=device).to(device)
    model.eval()

    dm = Robo2VLMDataModule(args.train_path, args.test_path, model.make_collator(),
                            batch_size=args.batch_size, num_workers=args.num_workers)

    examples = flatten_examples(dm.test_dataloader(), device, args.max_batches)
    print(f"[sampling] {len(examples)} examples x {len(STRATEGIES)} strategies "
          f"x {len(args.seeds)} seeds")

    rows = []
    for strategy in STRATEGIES:
        row = run_strategy(model, examples, strategy, args.seeds,
                           args.max_new_tokens, device)
        rows.append(row)
        print(f"  {row['strategy']:<14} acc={row['accuracy']:.3f} "
              f"valid={row['valid_label_rate']:.3f} "
              f"consist={row['self_consistency']:.3f} "
              f"support={row['mean_support']:.1f}")

    save(rows, args.out, len(examples), args.seeds)
    print(f"[sampling] -> {args.out}")