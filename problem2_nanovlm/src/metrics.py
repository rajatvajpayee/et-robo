import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, cohen_kappa_score, confusion_matrix)

"""
Cohen's Kappa:
Measures agreement after removing agreement expected by chance.
Example - model accuracy 0.80, chance agreement 0.60:
    kappa = (0.80 - 0.60) / (1 - 0.60) = 0.50
Higher kappa means the model performs substantially better than chance.

ECE (Expected Calibration Error):
"When the model says it is 90% confident, is it actually right 90% of the time?"
Predictions are grouped into confidence bins; ECE is the average gap between
confidence and accuracy in each bin, weighted by bin size.

Perplexity: how uncertain a probabilistic model is about the true next token/label

Probability mass outside the option labels:
how much probability the model puts on tokens that are not one of the option
numbers. Near zero means the output format has been fully learned, which is what
makes perplexity readable as task uncertainty rather than format confusion.
"""


def ece(probs, correct, n_bins=15):
    conf = probs.max(1)
    edges = np.linspace(0, 1, n_bins + 1)
    e, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not m.any():
            bins.append({"lo": float(lo), "n": 0, "conf": None, "acc": None})
            continue
        c, a = conf[m].mean(), correct[m].mean()
        e += m.mean() * abs(a - c)
        bins.append({"lo": float(lo), "n": int(m.sum()),
                     "conf": float(c), "acc": float(a)})
    return float(e), bins


def compute_all(y, p, probs, nll_sum=None, nll_tokens=None, outside_mass=None,
                n_classes=5, labels_names=None, n_bins=15):
    y, p = np.asarray(y), np.asarray(p)
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum(1, keepdims=True).clip(1e-12)
    correct = (y == p).astype(float)
    labels = list(range(n_classes))

    e, bins = ece(probs, correct, n_bins)

    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, p)),
        "macro_precision": float(precision_score(y, p, average="macro",
                                                 labels=labels, zero_division=0)),
        "macro_recall": float(recall_score(y, p, average="macro",
                                           labels=labels, zero_division=0)),
        "macro_f1": float(f1_score(y, p, average="macro", labels=labels,
                                   zero_division=0)),
        "cohens_kappa": float(cohen_kappa_score(y, p, labels=labels)),
        "ece": e,
        "mean_confidence": float(probs.max(1).mean()),
        "confusion_matrix": confusion_matrix(y, p, labels=labels).tolist(),
        "reliability_bins": bins,
    }

    if nll_sum is not None and nll_tokens:
        out["answer_nll"] = float(nll_sum / nll_tokens)
        out["perplexity"] = float(np.exp(nll_sum / nll_tokens))

    if outside_mass is not None:
        out["mean_prob_mass_outside_letters"] = float(np.mean(outside_mass))

    return out


def per_template(y, p, templates, n_classes=5, min_count=100):
    """Buckets smaller than min_count are dropped: kappa on a handful of rows
    has too much variance to interpret."""
    y, p = np.asarray(y), np.asarray(p)
    templates = np.asarray(templates)
    out = {}
    for t in sorted(set(templates.tolist())):
        m = templates == t
        if m.sum() < min_count:
            continue
        out[t] = {
            "n": int(m.sum()),
            "accuracy": float(accuracy_score(y[m], p[m])),
            "macro_f1": float(f1_score(y[m], p[m], average="macro",
                                       labels=list(range(n_classes)),
                                       zero_division=0)),
            "cohens_kappa": float(cohen_kappa_score(y[m], p[m],
                                                    labels=list(range(n_classes)))),
        }
    return out