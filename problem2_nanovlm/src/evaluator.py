import argparse
import json
import os
import re

import numpy as np
import torch
from tqdm import tqdm

import metrics as M
from prompts import LABELS
import plots

import setproctitle
setproctitle.setproctitle("evaluator.py")


def template_of(question, n_words=6):
    """group similar quesstions together with n_words for template analysis. """
    q = re.sub(r"[^a-z ]+", " ", str(question).lower())
    return " ".join(q.split()[:n_words])


def score_batch(model, g, device, label_ids, amp_dtype):
    images = g["images"].to(device).flatten(0, 1)
    input_ids = g["input_ids"].to(device)
    attention_mask = g["attention_mask"].to(device)
    labels = g["labels"].to(device)
    n_choices = g["n_choices"].to(device)
    b = labels.size(0)

    with torch.autocast(device, dtype=amp_dtype, enabled=device == "cuda"):
        logits = model(input_ids, images, attention_mask)

    
    ##Question...
    ##Answer: <- token before A  (g['answer_pos'] - 1)
    ##A
    pos = (g["answer_pos"] - 1).to(device)
    answer_logits = logits[torch.arange(b, device=device), pos].float()

    valid = torch.arange(len(LABELS), device=device)[None, :] < n_choices[:, None] # to select only valid choices. for example, if 4 choices were given in questions then 4 else 5
    label_logits = answer_logits.index_select(1, label_ids)
    probs = label_logits.masked_fill(~valid, float("-inf")).softmax(-1) # to make impossible option to -inf which makes softmax 0. 

    full = answer_logits.softmax(-1) # probs 
    outside = 1.0 - (full.index_select(1, label_ids) * valid).sum(-1) # 1 - probs(valid i.e 1,2,3,4 or 5)

    target = labels[:, 1:]
    keep = (target != -100) & torch.isin(target, label_ids)
    nll_sum, n_tokens = 0.0, 0
    if keep.any():
        selected = logits[:, :-1][keep].float().log_softmax(-1)
        token_nll = -selected.gather(1, target[keep][:, None]).squeeze(1)
        nll_sum = float(token_nll.sum())
        n_tokens = int(keep.sum())

    probs = probs.cpu().numpy()
    outside = outside.cpu()
    rows = [{
        "id": g["ids"][j],
        "template": template_of(g["questions"][j]),
        "gold": int(g["answer_idx"][j]),
        "pred": int(probs[j].argmax()),
        "probs": probs[j].tolist(),
        "n_choices": int(g["n_choices"][j]),
        "p_outside_labels": float(outside[j]),
    } for j in range(b)]

    return rows, nll_sum, n_tokens


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype=torch.bfloat16, max_batches=None):
    model.eval()
    label_ids = torch.tensor(model.label_ids, device=device)
    rows, nll, ntok = [], 0.0, 0

    for i, batch in enumerate(tqdm(loader)):
        if max_batches and i >= max_batches: # used it for initial training period. to break the evalaution mode for initial n_batches
            break
        for g in batch:
            group_rows, group_nll, group_tokens = score_batch(
                model, g, device, label_ids, amp_dtype)
            rows.extend(group_rows)
            nll += group_nll
            ntok += group_tokens

    y = np.array([r["gold"] for r in rows])
    p = np.array([r["pred"] for r in rows])
    probs = np.array([r["probs"] for r in rows])

    res = M.compute_all(y, p, probs,
                        nll_sum=nll, nll_tokens=ntok,
                        outside_mass=[r["p_outside_labels"] for r in rows],
                        n_classes=len(LABELS))
    res["per_template"] = M.per_template(
        y, p, [r["template"] for r in rows], len(LABELS))
    return res, rows


def save(res, rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(out_dir, "predictions.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    plots.eval_plots(res, rows, os.path.join(out_dir, "figures"))

    print(json.dumps({k: v for k, v in res.items()
                      if isinstance(v, (int, float))}, indent=2))


if __name__ == "__main__":
    from dataset import Robo2VLMDataModule
    from model import NanoVLM

    ap = argparse.ArgumentParser()
    ap.add_argument("--train-path", default="/home/rajat/scratch/et-robo/data/Robo2VLM-1_local/train")
    ap.add_argument("--test-path", default="/home/rajat/scratch/et-robo/data/Robo2VLM-1_local/test")
    ap.add_argument("--checkpoint", default=None,
                    help="save_pretrained directory; omit for the pre-trained baseline")
    ap.add_argument("--out", default="logs/eval")
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--max-batches", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NanoVLM(checkpoint=args.checkpoint, device=device).to(device)

    dm = Robo2VLMDataModule(args.train_path, args.test_path, model.make_collator(),
                            batch_size=args.batch_size, num_workers=args.num_workers)
    loader = {"test": dm.test_dataloader,
              "val": dm.val_dataloader,
              "train": dm.train_dataloader}[args.split]()

    res, rows = evaluate(model, loader, device, max_batches=args.max_batches)
    save(res, rows, args.out)
    print(f"[eval] -> {args.out}")