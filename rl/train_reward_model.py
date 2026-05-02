# coding=utf-8
import os
import sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast

from rl.reward_model import RewardModel, RewardModelConfig, init_reward_model_from_base


@dataclass
class RewardTrainConfig:
    tokenizer_dir: str = "./model_save/moe_pretrain/"
    base_model_dir: str = "./model_save/moe_pretrain"
    preference_file: str = "./data/rlaif_preferences.jsonl"
    output_dir: str = "./model_save/reward_model"

    seed: int = 23333
    mixed_precision: str = "bf16"

    max_seq_len: int = 512
    pooling: str = "mean"

    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    lr: float = 1e-4
    weight_decay: float = 0.0
    epochs: int = 1
    log_steps: int = 20
    save_steps: int = 200
    max_train_rows: int = 0


class PreferenceDataset(Dataset):
    def __init__(self, file: str, max_rows: int = 0) -> None:
        super().__init__()
        items = []
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
                if max_rows and len(items) >= int(max_rows):
                    break
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        obj = self.items[idx]
        return {"prompt": obj["prompt"], "chosen": obj["chosen"], "rejected": obj["rejected"]}


def collate_pref(batch: List[Dict[str, str]], tokenizer: PreTrainedTokenizerFast, max_seq_len: int) -> Dict[str, torch.Tensor]:
    prompts = [b["prompt"] for b in batch]
    chosen = [b["chosen"] for b in batch]
    rejected = [b["rejected"] for b in batch]

    chosen_txt = [p + c for p, c in zip(prompts, chosen)]
    rejected_txt = [p + r for p, r in zip(prompts, rejected)]

    enc_c = tokenizer(chosen_txt, padding=True, truncation=True, max_length=max_seq_len, return_tensors="pt", add_special_tokens=False)
    enc_r = tokenizer(rejected_txt, padding=True, truncation=True, max_length=max_seq_len, return_tensors="pt", add_special_tokens=False)

    return {
        "chosen_input_ids": enc_c["input_ids"],
        "chosen_attention_mask": enc_c["attention_mask"],
        "rejected_input_ids": enc_r["input_ids"],
        "rejected_attention_mask": enc_r["attention_mask"],
    }


def train_reward_model(cfg: Optional[RewardTrainConfig] = None) -> None:
    cfg = cfg or RewardTrainConfig()
    set_seed(cfg.seed)

    accelerator = Accelerator(mixed_precision=cfg.mixed_precision, gradient_accumulation_steps=cfg.gradient_accumulation_steps)
    is_main = accelerator.is_main_process

    tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.tokenizer_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = "[PAD]"

    rm, rm_cfg = init_reward_model_from_base(cfg.base_model_dir, pooling=cfg.pooling, map_location="cpu")
    dataset = PreferenceDataset(cfg.preference_file, max_rows=cfg.max_train_rows)

    dl = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_pref(b, tokenizer=tokenizer, max_seq_len=cfg.max_seq_len),
    )

    optimizer = torch.optim.AdamW(rm.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rm, optimizer, dl = accelerator.prepare(rm, optimizer, dl)

    if is_main:
        os.makedirs(cfg.output_dir, exist_ok=True)
        accelerator.print(f"[rm] dataset={len(dataset)} batch_size={cfg.batch_size} accum={cfg.gradient_accumulation_steps}")

    global_step = 0
    t0 = time.time()
    rm.train()
    for epoch in range(int(cfg.epochs)):
        for _, batch in enumerate(dl):
            with accelerator.accumulate(rm):
                s_c = rm(batch["chosen_input_ids"], batch["chosen_attention_mask"])
                s_r = rm(batch["rejected_input_ids"], batch["rejected_attention_mask"])
                loss = -F.logsigmoid(s_c - s_r).mean()

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1

                if is_main and global_step % int(cfg.log_steps) == 0:
                    dt = time.time() - t0
                    accelerator.print(f"[rm] step={global_step} loss={loss.item():.4f} time={dt:.1f}s")

                if is_main and global_step % int(cfg.save_steps) == 0:
                    save_dir = os.path.join(cfg.output_dir, f"step_{global_step}")
                    accelerator.print(f"[rm] saving to {save_dir}")
                    unwrapped: RewardModel = accelerator.unwrap_model(rm)
                    unwrapped.save_pretrained(save_dir, rm_cfg=RewardModelConfig(base_model_dir=rm_cfg.base_model_dir, pooling=rm_cfg.pooling))
                    tokenizer.save_pretrained(save_dir)

    if is_main:
        unwrapped: RewardModel = accelerator.unwrap_model(rm)
        unwrapped.save_pretrained(cfg.output_dir, rm_cfg=RewardModelConfig(base_model_dir=rm_cfg.base_model_dir, pooling=rm_cfg.pooling))
        tokenizer.save_pretrained(cfg.output_dir)
        accelerator.print(f"[rm] done. saved to {cfg.output_dir}")


if __name__ == "__main__":
    train_reward_model()
