import torch
import torch.nn as nn

import sys
sys.path.append("/vision01_scratch/rajat/et-robo/problem2_nanovlm/nanoVLM")

from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor, get_image_string

import setproctitle
setproctitle.setproctitle("VLM-base")


def build_prompt(question, choice):
    return f"""Question:
{question}

Candidate Answer:
{choice}

Is this candidate answer correct?

Answer:
"""


class NanoVLMClassifier(nn.Module):

    def __init__(self, checkpoint=None, hf_model="lusxvr/nanoVLM-230M-8k", dropout=0.1, device="cpu"):
        super().__init__()

        self.device = device
        self.vlm = VisionLanguageModel.from_pretrained(checkpoint if checkpoint else hf_model)

        for p in self.vlm.parameters():
            p.requires_grad = False

        self.tokenizer = get_tokenizer(
            self.vlm.cfg.lm_tokenizer,
            self.vlm.cfg.vlm_extra_tokens,
            self.vlm.cfg.lm_chat_template,
        )

        self.image_processor = get_image_processor(
            self.vlm.cfg.max_img_size,
            self.vlm.cfg.vit_img_size,
            getattr(self.vlm.cfg, "resize_to_max_side_len", False),
        )

        hidden_dim = self.vlm.decoder.cfg.lm_hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, input_ids, processed_image, attention_mask):

        hidden_states, _ = self.vlm(
            input_ids=input_ids,
            images=processed_image,
            attention_mask=attention_mask,
            return_hidden_states=True,
        )

        last_idx = attention_mask.sum(dim=1) - 1

        features = hidden_states[
            torch.arange(hidden_states.size(0), device=input_ids.device),
            last_idx,
        ]

        return self.classifier(features).squeeze(-1)

    def prepare_image(self, image):

        processed_image, split_ratio = self.image_processor(image)

        if (
            not hasattr(self.tokenizer, "global_image_token")
            and split_ratio[0] * split_ratio[1] == len(processed_image) - 1
        ):
            processed_image = processed_image[1:]

        image_string = get_image_string(
            self.tokenizer,
            [split_ratio],
            self.vlm.cfg.mp_image_token_length,
        )

        return processed_image.to(self.device), image_string

    def prepare_text(self, image_string, question, choice):

        prompt = image_string + build_prompt(question, choice)

        encoded = self.tokenizer.apply_chat_template(
            [[{"role": "user", "content": prompt}]],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        return (
            encoded["input_ids"].to(self.device),
            encoded["attention_mask"].to(self.device),
        )