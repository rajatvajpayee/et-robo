import torch

def apply_temperature(logits, T):
    if T is None or T == 1.0:
        return logits
    return logits / T


def apply_top_k(logits, k):
    if not k or k >= logits.size(-1):
        return logits
    kth = logits.topk(k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, float("-inf"))


def apply_top_p(logits, p):
    if p is None or p >= 1.0:
        return logits
    sorted_logits, idx = logits.sort(dim=-1, descending=True)
    probs = sorted_logits.softmax(-1)
    drop = probs.cumsum(-1) - probs >= p
    drop[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter_(-1, idx, sorted_logits)


def apply_min_p(logits, p):
    if not p:
        return logits
    probs = logits.softmax(-1)
    floor = p * probs.max(-1, keepdim=True).values
    return logits.masked_fill(probs < floor, float("-inf"))


def filter_logits(logits, mode="greedy", temperature=1.0, k=None, p=None):
    if mode == "greedy":
        return logits, True
    logits = apply_temperature(logits, temperature)
    if mode == "top_k":
        logits = apply_top_k(logits, k)
    elif mode == "top_p":
        logits = apply_top_p(logits, p)
    elif mode == "min_p":
        logits = apply_min_p(logits, p)
    elif mode != "temperature":
        raise ValueError(f"unknown mode: {mode}")
    return logits, False


def support_size(logits, **kw):
    filtered, greedy = filter_logits(logits.clone(), **kw)
    if greedy:
        return 1.0
    return float(torch.isfinite(filtered).sum(-1).float().mean())


@torch.no_grad()
def generate_one(model, input_ids, images, attention_mask, max_new_tokens=4,
                 eos_id=None, generator=None, **kw):
    ids, mask = input_ids.clone(), attention_mask.clone()
    entropies, supports = [], []

    for _ in range(max_new_tokens):
        logits = model(ids, images, mask)[:, -1, :].float()

        probs = logits.softmax(-1)
        entropies.append(float(-(probs * probs.clamp_min(1e-12).log()).sum()))
        supports.append(support_size(logits, **kw))

        filtered, greedy = filter_logits(logits, **kw)
        if greedy:
            next_id = filtered.argmax(-1)
        else:
            next_id = torch.multinomial(filtered.softmax(-1), 1,
                                        generator=generator).squeeze(-1)

        ids = torch.cat([ids, next_id[:, None]], dim=1)
        mask = torch.cat([mask, torch.ones_like(next_id)[:, None]], dim=1)

        if eos_id is not None and int(next_id) == eos_id:
            break

    generated = ids[:, input_ids.size(1):]
    return generated, sum(entropies) / len(entropies), sum(supports) / len(supports)