import argparse
import json
import math
import os
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prompts import LABELS
import setproctitle
setproctitle.setproctitle("gradcam.py")


EXAMPLE_IDS = [
    146, 535, 895, 971, 1357, 1402, 1678, 2018, 2520, 2546,
    2570, 2726, 2851, 2893, 2947, 2989, 2997, 3143, 3207, 3266,
    3304, 3306, 3485, 3775, 4140, 4368, 4677, 4798, 4869, 5060,
    5200, 5203, 5376, 5437, 5473, 5564, 5710, 5727, 5815, 5995,
] # the robot is tasked to pick. .

EXAMPLE_IDS = [3116, 3123, 3124, 3126, 3128, 3129, 3133, 3134, 3135, 3136, 3140, 3150, 3152, 3153, 3154, 3162, 3163, 3172, 3179, 3185, 3188, 3189, 3190, 3231, 3234, 3276, 3279, 3285, 3290, 3291, 3293, 3301, 3302, 3305, 3307, 3308, 3312, 3440, 3444, 3445, 3446, 3448, 3452,3455, 3459, 3461, 3462, 3464, 3465, 3466, 3470, 3472, 3474, 3477, 3478, 3480, 3481, 3484, 3487, 3489] # in the image from ext. 


@contextmanager
def record_attention():
    """Collect the attention probabilities of one forward pass.

    nanoVLM does not expose attention weights, and the fused kernels never
    build the probability matrix, so we temporarily wrap PyTorch's attention
    function and recompute the softmax ourselves.
    """
    original = F.scaled_dot_product_attention
    collected = []

    def wrapped(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
                scale=None, **kw):
        out = original(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                       is_causal=is_causal, scale=scale, **kw)
        with torch.no_grad():
            scale = scale or 1.0 / math.sqrt(q.size(-1))
            scores = (q.float() @ k.float().transpose(-2, -1)) * scale
            n_query, n_key = q.size(-2), k.size(-2)
            if is_causal:
                mask = torch.ones(n_query, n_key, dtype=torch.bool,
                                  device=q.device).tril(n_key - n_query)
                scores = scores.masked_fill(~mask, float("-inf"))
            collected.append(scores.softmax(-1).cpu())
        return out

    F.scaled_dot_product_attention = wrapped
    try:
        yield collected
    finally:
        F.scaled_dot_product_attention = original


def rollout(attentions):
    if not attentions:
        return None

    size = attentions[-1].size(-1)
    layers = [a for a in attentions
              if a.size(-1) == size and a.size(-2) == size]
    if not layers:
        return None
    layers = layers[-max(1, len(layers) // 4):]
    combined = sum(layer[0].mean(0) for layer in layers) / len(layers)
    return combined


def grad_cam(model, input_ids, images, attention_mask, pos, gold_id):
    """How much each image token changed the score of the correct answer.

    Takes the gradient of the gold answer's logit with respect to the image
    token embeddings, multiplies by the embeddings themselves, and keeps the
    positive part.
    """
    projector = None
    for name, module in model.vlm.named_modules():
        if "modality" in name.lower() or name.lower().endswith("mp"):
            projector = module
            break
    if projector is None:
        return None

    features = {}
    handle = projector.register_forward_hook(
        lambda m, inp, out: features.update(
            value=out[0] if isinstance(out, tuple) else out))
    try:
        model.vlm.zero_grad(set_to_none=True)
        logits = model(input_ids, images, attention_mask)

        image_tokens = features["value"]
        image_tokens.retain_grad()
        logits[0, pos, gold_id].backward()

        cam = (image_tokens.grad[0] * image_tokens[0]).sum(-1).relu()
        return cam.detach().cpu().numpy()
    finally:
        handle.remove()
        model.vlm.zero_grad(set_to_none=True)


def to_heatmap(values, height, width):
    """Lay a per-image-token vector back out on the patch grid."""
    side = int(round(math.sqrt(len(values))))
    grid = np.asarray(values[:side * side], float).reshape(side, side)
    grid = (grid - grid.min()) / max(grid.max() - grid.min(), 1e-9)
    return F.interpolate(torch.tensor(grid)[None, None], size=(height, width),
                         mode="bilinear", align_corners=False)[0, 0].numpy()


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_maps(image, rollout_map, cam_map, path, title):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    axes[0].imshow(image)
    axes[0].set_title("input", fontsize=9)
    for ax, heatmap, name in zip(axes[1:], [rollout_map, cam_map],
                                 ["attention rollout", "grad-CAM"]):
        ax.imshow(image)
        ax.imshow(heatmap, cmap="jet", alpha=0.55)
        ax.set_title(name, fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    save(fig, path)


def plot_attention_matrix(attention, n_image_tokens, input_ids, tokenizer,
                          path, title):
    """Image tokens (rows) against every position in the sequence (columns).

    Column 0 is blanked out. It is an attention sink -- the one token every
    other position can see -- and it takes so much of the weight that nothing
    else is visible on the same colour scale.
    """
    matrix = attention[:,:n_image_tokens].numpy().copy()
    matrix[:, 0] = 0.0
    ceiling = np.percentile(matrix[matrix > 0], 99) if (matrix > 0).any() else 1.0

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(matrix, cmap="magma", aspect="auto", vmin=0, vmax=ceiling)
    ax.axhline(n_image_tokens - 0.5, color="cyan", lw=0.9)

    ids = input_ids[0].tolist()
    ticks = list(range(n_image_tokens, min(len(ids), matrix.shape[1]), 4))
    if ticks:
        ax.set_xticks(ticks,
                      [tokenizer.decode([ids[i]]).replace("\n", "\\n").strip()[:10]
                       or str(ids[i]) for i in ticks],
                      rotation=90, fontsize=5)

    ax.set_ylabel("sequence position: image tokens, then prompt text", fontsize=8)
    ax.set_xlabel("image token", fontsize=8)
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, fraction=0.03)
    save(fig, path)


def spearman(a, b):
    """Rank correlation, so the two maps can be compared without assuming
    their scales match."""
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = math.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom else 0.0


def explain(model, collate, sample, device):
    batch = collate([sample])[0]
    images = batch["images"].to(device).flatten(0, 1)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    pos = int(batch["answer_pos"][0]) - 1
    gold_id = model.label_ids[int(batch["answer_idx"][0])]
    n_image_tokens = int((batch["input_ids"] == collate.image_token_id).sum())

    with record_attention() as collected, torch.no_grad():
        model(input_ids, images, attention_mask)
    attention = rollout(collected)

    rollout_row = attention[min(pos, attention.size(0) - 1), :n_image_tokens].numpy()
    cam = grad_cam(model, input_ids, images, attention_mask, pos, gold_id)

    return {
        "image": sample["image"].convert("RGB"),
        "attention": attention,
        "rollout_row": rollout_row,
        "cam": cam,
        "input_ids": batch["input_ids"],
        "n_image_tokens": n_image_tokens,
        "attention_on_image": float(rollout_row.sum()),
        "rollout_vs_gradcam": spearman(rollout_row, cam) if cam is not None else None,
    }


if __name__ == "__main__":
    from datasets import load_from_disk
    from model import NanoVLM

    parser = argparse.ArgumentParser()
    parser.add_argument("--test-path",
                        default="/home/rajat/scratch/et-robo/data/Robo2VLM-1_local/test")
    parser.add_argument("--checkpoint", default=None,
                        help="omit to run the pre-trained baseline")
    parser.add_argument("--out", default="logs/saliency")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NanoVLM(checkpoint=args.checkpoint, device=device).to(device)
    model.eval()
    collate = model.make_collator()
    dataset = load_from_disk(args.test_path)

    indices = [i for i in EXAMPLE_IDS if i < len(dataset)]
    print(f"[saliency] {len(indices)} examples")

    results = []
    for i in indices:
        sample = dataset[i]
        result = explain(model, collate, sample, device)

        height, width = result["image"].size[::-1]
        title = (f"{sample['question'][:70]}   "
                 f"gold = {LABELS[int(sample['correct_answer'])]}")

        plot_maps(result["image"],
                  to_heatmap(result["rollout_row"], height, width),
                  to_heatmap(result["cam"], height, width),
                  os.path.join(args.out, f"{i:04d}.png"), title)

        plot_attention_matrix(result["attention"], result["n_image_tokens"],
                              result["input_ids"], model.tokenizer,
                              os.path.join(args.out, "attention",
                                           f"{i:04d}_attention.png"), title)

        results.append({"index": i,
                        "attention_on_image": result["attention_on_image"],
                        "rollout_vs_gradcam": result["rollout_vs_gradcam"]})

    summary = {
        "n": len(results),
        "mean_attention_on_image":
            float(np.mean([r["attention_on_image"] for r in results])),
        "mean_rollout_vs_gradcam":
            float(np.mean([r["rollout_vs_gradcam"] for r in results
                           if r["rollout_vs_gradcam"] is not None])),
    }

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "saliency.json"), "w") as f:
        json.dump({"summary": summary, "per_example": results}, f, indent=2)

    print(summary)
    print(f"[saliency] -> {args.out}")