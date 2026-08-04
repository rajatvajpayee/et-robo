import csv
import os
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
import setproctitle
setproctitle.setproctitle("train.py")

class Trainer:

    def __init__(self, model, train_loader, val_loader, optimizer, device="cuda",
                 amp_dtype=torch.bfloat16, max_grad_norm=1.0,
                 log_dir="logs", run_name=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.amp_dtype = amp_dtype
        self.max_grad_norm = max_grad_norm
        self.best_acc = 0.0

        run_name = run_name or time.strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(log_dir, run_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.step_log = open(os.path.join(self.log_dir, "steps.csv"), "w", newline="")
        self.writer = csv.writer(self.step_log)
        self.writer.writerow(["step", "train_loss", "train_acc", "val_loss",
                              "val_ppl", "val_acc", "grad_norm", "seconds"])
        self.t0 = time.time()
        self.ckpt_dir = os.path.join(self.log_dir, "ckpt")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.best_val_acc = 0.0

    def _to_device(self, group):
        return {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in group.items()}

    def _step(self, group):
        g = self._to_device(group)
        images = g["images"].flatten(0, 1)

        with torch.autocast(self.device, dtype=self.amp_dtype,
                            enabled=self.device == "cuda"):
            logits = self.model(g["input_ids"], images, g["attention_mask"])
        
        labels = g["labels"]
        # logits = logits.float()
        # loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                            #    labels[:, 1:].reshape(-1), ignore_index=-100)
        shift_labels = labels[:, 1:]
        mask = shift_labels != -100
        loss = F.cross_entropy(logits[:, :-1][mask].float(), shift_labels[mask])


        n_tokens = int((labels[:, 1:] != -100).sum())
        b = labels.size(0)
        pos = g["answer_pos"] - 1
        label_ids = torch.tensor(self.model.label_ids, device=logits.device)
        sel = logits[torch.arange(b, device=logits.device), pos].index_select(1, label_ids)
        nc = g["n_choices"]
        sel = sel.masked_fill(
            torch.arange(sel.size(1), device=logits.device)[None, :] >= nc[:, None],
            float("-inf"))
        correct = (sel.argmax(-1) == g["answer_idx"]).sum().item()
        return loss, correct, b, n_tokens

    def train_epoch(self, val_every=25, val_batches=20, max_steps=None):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        run_loss, run_correct, run_total = 0.0, 0, 0

        for step, batch in enumerate(tqdm(self.train_loader), 1):
            self.optimizer.zero_grad(set_to_none=True)
            n_batch = sum(g["labels"].size(0) for g in batch)

            for group in batch:
                loss, c, n, _ = self._step(group)
                (loss * n / n_batch).backward()
                total_loss += loss.item() * n
                correct += c
                total += n
                run_loss += loss.item() * n
                run_correct += c
                run_total += n

            gn = torch.nn.utils.clip_grad_norm_(
                self.model.trainable_parameters(), self.max_grad_norm)
            self.optimizer.step()

            if val_every and step % val_every == 0:
                vl, vp, va = self.validate(max_batches=val_batches)
                tl, ta = run_loss / run_total, run_correct / run_total
                print(f"step {step} | train {tl:.4f}/{ta:.4f} "
                      f"| val {vl:.4f}/ppl {vp:.2f}/{va:.4f}")
                self.writer.writerow([step, f"{tl:.4f}", f"{ta:.4f}", f"{vl:.4f}",
                                      f"{vp:.4f}", f"{va:.4f}", f"{float(gn):.3f}",
                                      f"{time.time()-self.t0:.1f}"])
                self.step_log.flush()
                run_loss, run_correct, run_total = 0.0, 0, 0
                self.model.train()
                self._save(f"step{step}", step, {"loss": vl, "ppl": vp, "acc": va})
                if va > self.best_val_acc:
                    self.best_val_acc = va
                    self._save("best", step, {"loss": vl, "ppl": vp, "acc": va})

            if max_steps and step >= max_steps:
                break

        return total_loss / total, correct / total

    @torch.no_grad()
    def validate(self, max_batches=None):
        self.model.eval()
        nll, ntok, correct, total = 0.0, 0, 0, 0

        for i, batch in enumerate(tqdm(self.val_loader, leave=False)):
            if max_batches and i >= max_batches:
                break
            for group in batch:
                loss, c, n, nt = self._step(group)
                nll += loss.item() * nt
                ntok += nt
                correct += c
                total += n

        mean_nll = nll / max(ntok, 1)
        return mean_nll, float(torch.tensor(mean_nll).exp()), correct / max(total, 1)

    def fit(self, epochs, val_every=25, val_batches=20, max_steps=None):
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(val_every, val_batches, max_steps)
            val_loss, val_ppl, val_acc = self.validate()

            print(f"Epoch {epoch+1}/{epochs} | Train Loss {train_loss:.4f} | "
                  f"Train Acc {train_acc:.4f} | Val Loss {val_loss:.4f} | "
                  f"Val PPL {val_ppl:.3f} | Val Acc {val_acc:.4f}")
            self.writer.writerow([f"epoch{epoch+1}", f"{train_loss:.4f}",
                                  f"{train_acc:.4f}", f"{val_loss:.4f}",
                                  f"{val_ppl:.4f}", f"{val_acc:.4f}", "",
                                  f"{time.time()-self.t0:.1f}"])
            self.step_log.flush()

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.model.vlm.save_pretrained(os.path.join(self.log_dir, "best"))

        self.step_log.close()

    def _save(self, name, step, val=None):
        torch.save({
            "step": step,
            "model": self.model.vlm.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "val": val,
        }, os.path.join(self.ckpt_dir, f"{name}.pth"))


if __name__ == "__main__":

    import argparse
    from dataset import Robo2VLMDataModule
    from model import NanoVLM

    ap = argparse.ArgumentParser()
    ap.add_argument("--train-path", default="/home/rajat/scratch/et-robo/data/Robo2VLM-1_local/train")
    ap.add_argument("--test-path", default="/home/rajat/scratch/et-robo/data/Robo2VLM-1_local/test")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=36)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1.5e-5)
    ap.add_argument("--val-every", type=int, default=3000)
    ap.add_argument("--val-batches", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--unfreeze-vision", action="store_true")
    ap.add_argument("--run-name", default=None)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NanoVLM(freeze_vision=not a.unfreeze_vision, device=device)

    dm = Robo2VLMDataModule(
        train_path=a.train_path,
        test_path=a.test_path,
        collate_fn=model.make_collator(),
        batch_size=a.batch_size,
        val_split=0.1,
        num_workers=a.num_workers,
    )

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=a.lr,
                                  weight_decay=1e-2)

    trainer = Trainer(model, dm.train_dataloader(), dm.val_dataloader(),
                      optimizer, device=device, run_name=a.run_name)
    trainer.fit(a.epochs, val_every=a.val_every, val_batches=a.val_batches,
                max_steps=a.max_steps)
