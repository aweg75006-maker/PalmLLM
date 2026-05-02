# coding=utf-8
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from pyarrow import parquet as pq
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast

from model.moe_decoder_lm import MoeDecoderConfig, MoeDecoderLM


@dataclass
class MoePretrainConfig:
    tokenizer_dir: str = "./model_save/"
    train_parquet: str = "./data/my_train_dataset_3k.parquet" # 目前的 3k 数据更适合做微调而不是预训练。
    output_dir: str = "./model_save/moe_pretrain"

    seed: int = 23333
    mixed_precision: str = "bf16"  # "no" | "fp16" | "bf16"

    # model
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    n_experts: int = 8
    top_k: int = 2
    capacity_factor: float = 1.25
    dropout: float = 0.0
    max_seq_len: int = 512
    attn_chunk_size: int = 0

    # train
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    lr: float = 2e-4
    weight_decay: float = 0.01
    epochs: int = 1
    log_steps: int = 20
    save_steps: int = 200
    max_train_rows: int = 0  # 0 => all rows
    
    # MoE auxiliary loss weight
    aux_loss_weight: float = 0.01  # 辅助损失权重，鼓励负载均衡


class CausalParquetDataset(Dataset):
    """
    Reads parquet with columns: prompt, response.
    Builds causal LM samples by concatenating: prompt + response + [EOS].
    """

    def __init__(
        self,
        parquet_file: str,
        tokenizer: PreTrainedTokenizerFast,
        max_seq_len: int,
        max_rows: int = 0,
    ) -> None:
        super().__init__()
        table = pq.read_table(parquet_file)
        if max_rows and max_rows > 0:
            table = table.slice(0, min(int(max_rows), table.num_rows))
        self.prompt = table["prompt"]
        self.response = table["response"]
        self.length = table.num_rows
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)

    def __len__(self) -> int:
        return int(self.length)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        p = self.prompt[idx].as_py()
        r = self.response[idx].as_py()
        txt = f"{p}{r}[EOS]"
        ids = self.tokenizer.encode(txt, add_special_tokens=False, truncation=True, max_length=self.max_seq_len)
        return {"input_ids": ids}


def collate_causal(batch: List[Dict[str, List[int]]], pad_id: int, max_seq_len: int) -> Dict[str, torch.Tensor]:
    lens = [len(x["input_ids"]) for x in batch]
    max_len = min(max(lens), int(max_seq_len))
    input_ids = np.full((len(batch), max_len), pad_id, dtype=np.int64)
    attn_mask = np.zeros((len(batch), max_len), dtype=np.int64)
    for i, item in enumerate(batch):
        ids = item["input_ids"][:max_len]
        input_ids[i, : len(ids)] = np.asarray(ids, dtype=np.int64)
        attn_mask[i, : len(ids)] = 1
    labels = input_ids.copy()
    labels[attn_mask == 0] = -100
    return {
        "input_ids": torch.from_numpy(input_ids),
        "attention_mask": torch.from_numpy(attn_mask),
        "labels": torch.from_numpy(labels),
    }


def pretrain_moe(cfg: Optional[MoePretrainConfig] = None) -> None:
    cfg = cfg or MoePretrainConfig()
    set_seed(cfg.seed)

    accelerator = Accelerator(mixed_precision=cfg.mixed_precision, gradient_accumulation_steps=cfg.gradient_accumulation_steps)
    is_main = accelerator.is_main_process

    tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.tokenizer_dir)
    if tokenizer.pad_token_id is None:
        # match repo convention
        tokenizer.pad_token = "[PAD]"

    model_cfg = MoeDecoderConfig(
        vocab_size=len(tokenizer),
        max_seq_len=cfg.max_seq_len,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        n_experts=cfg.n_experts,
        top_k=cfg.top_k,
        d_ff=cfg.d_ff,
        capacity_factor=cfg.capacity_factor,
        dropout=cfg.dropout,
        attn_chunk_size=cfg.attn_chunk_size,
    )
    model = MoeDecoderLM(model_cfg)

    dataset = CausalParquetDataset(
        parquet_file=cfg.train_parquet,
        tokenizer=tokenizer,
        max_seq_len=cfg.max_seq_len,
        max_rows=cfg.max_train_rows,
    )

    dl = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_causal(b, pad_id=tokenizer.pad_token_id, max_seq_len=cfg.max_seq_len),
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    model, optimizer, dl = accelerator.prepare(model, optimizer, dl)

    if is_main:
        os.makedirs(cfg.output_dir, exist_ok=True)
        accelerator.print(f"[moe_pretrain] dataset={len(dataset)} batch_size={cfg.batch_size} accum={cfg.gradient_accumulation_steps}")

    global_step = 0
    t0 = time.time()
    model.train()
    for epoch in range(int(cfg.epochs)):
        for step, batch in enumerate(dl):
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                labels = batch["labels"]

                logits, _, aux_loss = model(input_ids)
                # next-token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                main_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
                
                # 添加辅助损失（如果存在）
                if aux_loss is not None:
                    loss = main_loss + cfg.aux_loss_weight * aux_loss
                else:
                    loss = main_loss

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1

                if is_main and (global_step % int(cfg.log_steps) == 0):
                    dt = time.time() - t0
                    log_msg = f"[moe_pretrain] step={global_step} loss={loss.item():.4f} main_loss={main_loss.item():.4f}"
                    if aux_loss is not None:
                        log_msg += f" aux_loss={aux_loss.item():.4f}"
                    log_msg += f" time={dt:.1f}s"
                    accelerator.print(log_msg)

                if is_main and (global_step % int(cfg.save_steps) == 0):
                    save_dir = os.path.join(cfg.output_dir, f"step_{global_step}")
                    accelerator.print(f"[moe_pretrain] saving to {save_dir}")
                    unwrapped = accelerator.unwrap_model(model)
                    unwrapped.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)

    if is_main:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)
        accelerator.print(f"[moe_pretrain] done. saved to {cfg.output_dir}")


if __name__ == "__main__":
    # Example:
    #   python moe_pretrain.py
    pretrain_moe()

