"""
Data loading utilities for the Robo2VLM dataset.

This module loads the training and test datasets, creates a stratified
train/validation split, and provides DataLoaders for each split.
Instead of performing preprocessing inside a custom Dataset, all sample
processing is handled by the collator at batch time. 

The collator processesimages, constructs multimodal prompts, 
tokenizes the input, generates training labels, pads variable-length sequences 
and groups samples with the same image
split ratio into mini-batches compatible with the vision-language model.
"""

import ast
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from prompts import build_prompt, build_answer


class Collator:
    def __init__(self, tokenizer, image_processor, mp_image_token_length,
                 image_token, get_image_string, max_len=None):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length
        self.image_token = image_token
        self.get_image_string = get_image_string
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None \
            else tokenizer.eos_token_id
        self.max_len = max_len

    def _image(self, image):
        # split_ratio is the grid. 
        processed_image, split_ratio = self.image_processor(image) # process the image
        
        # Generates placeholder for image tokens. 64 image tokens. 
        image_string = self.get_image_string(
            self.tokenizer, [split_ratio], self.mp_image_token_length
        )
        return processed_image, image_string, split_ratio

    def _encode(self, prompt):
        ids = self.tokenizer.apply_chat_template(
            [[{"role": "user", "content": prompt}]],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors=None,
        )["input_ids"]
        return ids[0] if isinstance(ids[0], list) else ids

    def __call__(self, batch):
        by_ratio = {}
        for sample in batch:
            image, image_string, split_ratio = self._image(sample["image"])
            choices = ast.literal_eval(sample["choices"])
            answer_idx = int(sample["correct_answer"])
            prompt_ids = self._encode(image_string + build_prompt(sample["question"], choices))
            answer_ids = self.tokenizer.encode(build_answer(answer_idx),
                                               add_special_tokens=False)
            by_ratio.setdefault(tuple(split_ratio), []).append(
                (image, prompt_ids, answer_ids, answer_idx,
                 sample.get("id", ""), len(choices), sample["question"])
            )

        groups = []
        for items in by_ratio.values():
            images = torch.stack([i[0] for i in items])
            seqs = [i[1] + i[2] for i in items] # question prompt + answer 
            L = max(len(s) for s in seqs)
            n = len(items)
            input_ids = torch.full((n, L), self.pad_id, dtype=torch.long)
            attention_mask = torch.zeros((n, L), dtype=torch.long)
            labels = torch.full((n, L), -100, dtype=torch.long)
            for i, (s, item) in enumerate(zip(seqs, items)):
                input_ids[i, :len(s)] = torch.tensor(s)
                attention_mask[i, :len(s)] = 1
                labels[i, len(item[1]):len(s)] = torch.tensor(item[2])

            groups.append({
                "images": images,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "answer_pos": torch.tensor([len(i[1]) + 1 for i in items]),
                "answer_idx": torch.tensor([i[3] for i in items], dtype=torch.long),
                "n_choices": torch.tensor([i[5] for i in items]),
                "ids": [i[4] for i in items],
                "questions": [i[6] for i in items],
            })
        return groups


class Robo2VLMDataModule:
    def __init__(self, train_path, test_path, collate_fn, batch_size=8,
                 val_split=0.1, num_workers=4, seed=42):
        train_dataset = load_from_disk(train_path)
        test_dataset = load_from_disk(test_path)

        idx = list(range(len(train_dataset)))
        labels = train_dataset["correct_answer"]

        # To prepare the validation split with same distribution, I used stratify. 
        train_idx, val_idx = train_test_split(
            idx, test_size=val_split, random_state=seed, stratify=labels,
        )

        self.train_dataset = train_dataset.select(train_idx)
        self.val_dataset = train_dataset.select(val_idx)
        self.test_dataset = test_dataset

        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _loader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, False)
