import torch
import torch.nn as nn

import sys
sys.path.append("/vision01_scratch/rajat/et-robo/problem2_nanovlm_noMLP/nanoVLM")

from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor, get_image_string

from prompts import LABELS


class NanoVLM(nn.Module):

    def __init__(self, checkpoint=None, hf_model="lusxvr/nanoVLM-230M-8k",
                 freeze_vision=True, device="cpu"):
        super().__init__()

        self.device = device

        # Loads nanoVLM model
        self.vlm = VisionLanguageModel.from_pretrained(checkpoint if checkpoint else hf_model)

        if freeze_vision: # Freezing the vision encoder 
            for p in self.vlm.vision_encoder.parameters():
                p.requires_grad = False

        self.tokenizer = get_tokenizer(
            self.vlm.cfg.lm_tokenizer, # "HuggingFaceTB/SmolLM2-360M-Instruct"
            self.vlm.cfg.vlm_extra_tokens, # add vision tokens in vocab. 
            self.vlm.cfg.lm_chat_template, # Register chat template
        )

        ## What does it do?
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token # |EOS| token 

        
        # self.image_processor = get_image_processor(
        #     self.vlm.cfg.max_img_size, # this is 2048, but it is waste in this case, as the original image size is (320,256) upscaling it doesnt add more information. So, it is just waste of compute. hence tried but commented it later.
        #     self.vlm.cfg.vit_img_size,
        #     getattr(self.vlm.cfg, "resize_to_max_side_len", False),
        # )
        self.image_processor = get_image_processor(
            self.vlm.cfg.vit_img_size, # our image size is (320,256) -> resize to (512,512) -> create patches of size (512,512). So, here its just one
            self.vlm.cfg.vit_img_size,
            getattr(self.vlm.cfg, "resize_to_max_side_len", False),
        )
        self.label_ids = self._label_ids()

    def _label_ids(self):
        ids = [self.tokenizer.encode(l, add_special_tokens=False)[0] for l in LABELS]
        assert len(set(ids)) == len(LABELS), ids
        return ids

    def make_collator(self):
        from dataset import Collator
        return Collator(
            self.tokenizer,
            self.image_processor,
            self.vlm.cfg.mp_image_token_length,
            self.vlm.cfg.vlm_extra_tokens["image_token"],
            get_image_string,
        )

    def forward(self, input_ids, images, attention_mask):
        out = self.vlm(
            input_ids=input_ids,
            images=images,
            attention_mask=attention_mask,
        )
        h = out[0] if isinstance(out, (tuple, list)) else out
        if not self.vlm.decoder.lm_use_tokens: # Pass it thorough head. 
            '''In original implmentation, lm_use_token is False, which basically means 
            to not train head directly. Rather its trained indirectly
            Original nanoVLM snippet
            self.head = nn.Linear(cfg.lm_hidden_dim, cfg.lm_vocab_size, bias=False)
            if self.lm_tie_weights:
                self.head.weight = self.token_embedding.weight

            In this implemnetation,  we are passing through the head to get the final output. 
            '''
            h = self.vlm.decoder.head(h)
        return h

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

if __name__ == "__main__":
    m = NanoVLM(device="cpu")

    print("--- readout ---")
    print("lm_use_tokens :", m.vlm.decoder.lm_use_tokens)
    print("cfg vocab     :", m.vlm.cfg.lm_vocab_size)
    print("head weight   :", tuple(m.vlm.decoder.head.weight.shape))
    print("tokenizer len :", len(m.tokenizer))

    print("--- labels ---")
    print("label_ids     :", m.label_ids)
    print("decoded       :", [m.tokenizer.decode([i]) for i in m.label_ids])
    print("max label id  :", max(m.label_ids), "< vocab:",
          max(m.label_ids) < m.vlm.decoder.head.weight.shape[0])

    print("--- params ---")
    tr = sum(p.numel() for p in m.trainable_parameters())
    tot = sum(p.numel() for p in m.parameters())
    print(f"trainable     : {tr/1e6:.1f}M / {tot/1e6:.1f}M")
    print("head trainable:", m.vlm.decoder.head.weight.requires_grad)

    print("--- image ---")
    print("vit_img_size  :", m.vlm.cfg.vit_img_size)
    print("mp_tokens/tile:", m.vlm.cfg.mp_image_token_length)

    print("--- lm_tie_weights ---")
    print(m.vlm.cfg.lm_tie_weights) #ties the head with the token embeddings. 
    print(m.vlm.decoder.head.weight is m.vlm.decoder.token_embedding.weight)