# coding=utf-8
import os
import sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import json
import random
import time
from dataclasses import dataclass
from typing import Optional

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import PreTrainedTokenizerFast

from model.moe_decoder_lm import MoeDecoderLM
from rl.reward_model import RewardModel
from rl.rl_utils import logprob_of_suffix, scalar_logprob_of_suffix


@dataclass
class GrpoTrainConfig:
    tokenizer_dir: str = "./model_save/moe_pretrain"
    policy_dir: str = "./model_save/moe_pretrain"
    reward_model_dir: str = "./model_save/reward_model"
    # 一次生成多个回答（一个 group），用 reward model 给每个回答打分 计算“相对优势”：
    prompts_file: str = "./data/rlaif_prompts.jsonl"
    output_dir: str = "./model_save/grpo"

    seed: int = 23333
    mixed_precision: str = "bf16"

    # generation
    max_new_tokens: int = 128
    temperature: float = 0.9
    top_k: int = 50

    # GRPO-ish (group-relative PPO)
    updates: int = 10
    batch_size: int = 2
    group_size: int = 4
    lr: float = 2e-5
    weight_decay: float = 0.0
    clip_range: float = 0.2
    kl_beta: float = 0.02


def _iter_prompts(file: str) -> list[str]:
    prompts = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line)["prompt"])
    return prompts


def train_grpo(cfg: Optional[GrpoTrainConfig] = None) -> None:
    cfg = cfg or GrpoTrainConfig()
    set_seed(cfg.seed)
    random.seed(cfg.seed)

    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    is_main = accelerator.is_main_process

    tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.tokenizer_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = "[PAD]"

    policy = MoeDecoderLM.from_pretrained(cfg.policy_dir, map_location="cpu")
    ref = MoeDecoderLM.from_pretrained(cfg.policy_dir, map_location="cpu")
    rm = RewardModel.from_pretrained(cfg.reward_model_dir, map_location="cpu")

    rm.eval()
    for p in rm.parameters():
        p.requires_grad = False
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    device = accelerator.device
    policy.to(device)
    ref.to(device)
    rm.to(device)

    prompts = _iter_prompts(cfg.prompts_file)
    if is_main:
        os.makedirs(cfg.output_dir, exist_ok=True)
        accelerator.print(f"[grpo] prompts={len(prompts)} updates={cfg.updates} batch={cfg.batch_size} group={cfg.group_size}")

    t0 = time.time()
    policy.train()
    for update in range(int(cfg.updates)):
        batch_prompts = random.sample(prompts, k=min(int(cfg.batch_size), len(prompts)))
        samples = []

        for prompt in batch_prompts:
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            prompt_len = len(prompt_ids)
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

            group = []
            for gi in range(int(cfg.group_size)):
                with torch.no_grad():
                    torch.manual_seed(cfg.seed + update * 1000 + gi)
                    full_ids = policy.generate(
                        input_ids=input_ids,
                        max_new_tokens=cfg.max_new_tokens,
                        temperature=cfg.temperature,
                        top_k=cfg.top_k,
                        do_sample=True,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    attn_mask = torch.ones_like(full_ids, dtype=torch.long)
                    old_logp = scalar_logprob_of_suffix(policy, full_ids, prompt_len=prompt_len)
                    ref_logp = scalar_logprob_of_suffix(ref, full_ids, prompt_len=prompt_len)
                    reward = float(rm(full_ids, attn_mask).detach().cpu().item())
                    reward = reward - cfg.kl_beta * float(old_logp - ref_logp)

                # 循环结束后，group 列表里存放了针对同一个 Prompt 的多个不同采样结果及其对应的奖励分数
                group.append(
                    {
                        "prompt_len": prompt_len,
                        "input_ids": full_ids,
                        "attention_mask": torch.ones_like(full_ids, dtype=torch.long),
                        "old_logp": float(old_logp),
                        "reward": float(reward),
                    }
                )

            # 基准线（Baseline）的计算：求平均
            #
            mean_r = sum(x["reward"] for x in group) / max(1, len(group))
            for x in group:
                x["adv"] = float(x["reward"] - mean_r)  # group-relative (decentralized)
                samples.append(x)

        random.shuffle(samples)
        for s in samples:
            input_ids = s["input_ids"]
            old_logp = torch.tensor(s["old_logp"], device=device, dtype=torch.float32)
            adv = torch.tensor(s["adv"], device=device, dtype=torch.float32)

            new_logp = logprob_of_suffix(policy, input_ids, prompt_len=s["prompt_len"])
            ratio = torch.exp(new_logp - old_logp)
            clipped = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range)
            loss = -torch.min(ratio * adv, clipped * adv)

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            optimizer.step()

        if is_main:
            dt = time.time() - t0
            avg_r = sum(x["reward"] for x in samples) / max(1, len(samples))
            accelerator.print(f"[grpo] update={update+1}/{cfg.updates} avg_reward={avg_r:.4f} time={dt:.1f}s")

    if is_main:
        save_dir = cfg.output_dir
        accelerator.print(f"[grpo] saving to {save_dir}")
        policy.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        accelerator.print("[grpo] done")


if __name__ == "__main__":
    train_grpo()
