"""Fully written by GPT to compare the results when we give metrics.json as input. This file is for ease. 
    Give two jsons and print the resutls together. NO manual effort to look and get the value. 
"""
import argparse
import csv
import json
import os

HEADLINE = ["accuracy", "macro_f1", "cohens_kappa", "perplexity", "answer_nll",
            "ece", "mean_confidence", "mean_prob_mass_outside_letters"]
LOWER_BETTER = {"perplexity", "answer_nll", "ece", "mean_prob_mass_outside_letters"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--finetuned", required=True)
    ap.add_argument("--out", default="logs/comparison")
    a = ap.parse_args()

    B = json.load(open(os.path.join(a.baseline, "metrics.json")))
    F = json.load(open(os.path.join(a.finetuned, "metrics.json")))
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for k in HEADLINE:
        if k not in B and k not in F:
            continue
        b, f = B.get(k, float("nan")), F.get(k, float("nan"))
        rows.append({"metric": k, "baseline": b, "finetuned": f, "delta": f - b,
                     "direction": "lower" if k in LOWER_BETTER else "higher"})

    tpl = []
    bt, ft = B.get("per_template", {}), F.get("per_template", {})
    for k in sorted(set(bt) | set(ft)):
        b = bt.get(k, {}).get("accuracy", float("nan"))
        f = ft.get(k, {}).get("accuracy", float("nan"))
        tpl.append({"template": k, "n": ft.get(k, bt.get(k, {})).get("n", 0),
                    "baseline_acc": b, "finetuned_acc": f, "delta": f - b})
    tpl.sort(key=lambda r: -(r["delta"] if r["delta"] == r["delta"] else -9))

    with open(os.path.join(a.out, "comparison.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
    if tpl:
        with open(os.path.join(a.out, "per_template.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, list(tpl[0])); w.writeheader(); w.writerows(tpl)

    md = ["| metric | baseline | fine-tuned | delta | better |", "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['metric']} | {r['baseline']:.4f} | {r['finetuned']:.4f} | "
                  f"{r['delta']:+.4f} | {r['direction']} |")
    if tpl:
        md += ["", "| template | n | baseline | fine-tuned | delta |",
               "|---|---|---|---|---|"]
        for r in tpl:
            md.append(f"| {r['template']} | {r['n']} | {r['baseline_acc']:.3f} | "
                      f"{r['finetuned_acc']:.3f} | {r['delta']:+.3f} |")
    open(os.path.join(a.out, "comparison.md"), "w").write("\n".join(md))

    import plots
    plots.baseline_vs_ft(B, F, os.path.join(a.out, "figures", "baseline_vs_ft.png"))
    plots.per_template(F, os.path.join(a.out, "figures", "per_template.png"), baseline=B)
    print("\n".join(md[:2 + len(rows)]))


if __name__ == "__main__":
    main()