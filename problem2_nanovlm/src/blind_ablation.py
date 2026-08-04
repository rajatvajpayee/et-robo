"""Ablation: how much of the accuracy survives with the image removed?"""
import argparse, json, os
import torch

from evaluator import evaluate
from dataset import Robo2VLMDataModule
from model import NanoVLM


class BlankImageCollator:
    """Wraps the real collator and replaces pixels with a constant."""

    def __init__(self, collator, mode="zeros"):
        self.collator = collator
        self.mode = mode
        self.image_token_id = collator.image_token_id

    def __call__(self, batch):
        groups = self.collator(batch) # get the batch from existing collator

        ## Below code to make the images zeros/noise. 
        ### For this experiment, I have kept zeros. Idea is that I wanted to test 
        ### whether with no visual signal. how model performs. Basically to understandm
        ### the mirage - https://arxiv.org/abs/2603.21687
        for g in groups:
            if self.mode == "zeros":
                g["images"] = torch.zeros_like(g["images"])
            else:
                g["images"] = torch.randn_like(g["images"])
        return groups


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-path", default="/home/rajat/scratch/et-robo/data/Robo2VLM-1_local/test")
    ap.add_argument("--train-path", default="/home/rajat/scratch/et-robo/data/Robo2VLM-1_local/train")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="../logs/withDecoderHead/eval/epoch4/ablation_blank")
    ap.add_argument("--mode", default="zeros", choices=["zeros", "noise"])
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NanoVLM(checkpoint=args.checkpoint, device=device).to(device)

    dm = Robo2VLMDataModule(args.train_path, args.test_path,
                            BlankImageCollator(model.make_collator(), args.mode),
                            batch_size=args.batch_size, num_workers=8)

    res, rows = evaluate(model, dm.test_dataloader(), device)
    os.makedirs(args.out, exist_ok=True)
    json.dump(res, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    print(f"accuracy {res['accuracy']:.4f}  kappa {res['cohens_kappa']:.4f}")