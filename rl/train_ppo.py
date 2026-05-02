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
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import PreTrainedTokenizerFast

from model.moe_decoder_lm import MoeDecoderLM
from rl.ppo_policy import PpoPolicyConfig, PolicyWithValueHead, init_ppo_policy_from_base
from rl.reward_model import RewardModel
from rl.rl_utils import logprob_of_suffix, scalar_logprob_of_suffix


@dataclass
class PpoTrainConfig:
    tokenizer_dir: str = "./model_save/moe_pretrain"
    policy_base_dir: str = "./model_save/moe_pretrain"
    reward_model_dir: str = "./model_save/reward_model"
    prompts_file: str = "./data/rlaif_prompts.jsonl"
    output_dir: str = "./model_save/ppo"

    seed: int = 23333
    mixed_precision: str = "bf16"

    # generation
    max_new_tokens: int = 128
    temperature: float = 0.9
    top_k: int = 50

    # PPO
    updates: int = 10
    batch_size: int = 4
    ppo_epochs: int = 1
    lr: float = 2e-5
    weight_decay: float = 0.0
    clip_range: float = 0.2
    vf_coef: float = 0.5
    kl_beta: float = 0.02
    pooling: str = "mean"


def _iter_prompts(file: str) -> list[str]:
    prompts = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line)["prompt"])
    return prompts


def train_ppo(cfg: Optional[PpoTrainConfig] = None) -> None:
    cfg = cfg or PpoTrainConfig()
    set_seed(cfg.seed)
    random.seed(cfg.seed)

    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    is_main = accelerator.is_main_process

    tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.tokenizer_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = "[PAD]"

    # """
    #     policy 是 PolicyWithValueHead
    #     policy.parameters() 包含两部分 ： base LM（策略网络） + value head（价值网络）
    # """

    policy, policy_cfg = init_ppo_policy_from_base(cfg.policy_base_dir, pooling=cfg.pooling, map_location="cpu")
    ref = MoeDecoderLM.from_pretrained(cfg.policy_base_dir, map_location="cpu")
    rm = RewardModel.from_pretrained(cfg.reward_model_dir, map_location="cpu")

    # freeze RM + ref
    rm.eval()
    for p in rm.parameters():
        p.requires_grad = False
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    # 冻结 SFT 模型的 奖励模型（裁判）不参与训练 ， 仅策略模型参与训练
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    device = accelerator.device
    policy.to(device)
    ref.to(device)
    rm.to(device)

    prompts = _iter_prompts(cfg.prompts_file)
    if is_main:
        os.makedirs(cfg.output_dir, exist_ok=True)
        accelerator.print(f"[ppo] prompts={len(prompts)} updates={cfg.updates} batch={cfg.batch_size}")

    t0 = time.time()
    policy.train()
    for update in range(int(cfg.updates)):
        batch_prompts = random.sample(prompts, k=min(int(cfg.batch_size), len(prompts)))

        trajectories = []
        for prompt in batch_prompts:
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            prompt_len = len(prompt_ids)
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

            with torch.no_grad():
                full_ids = policy.base.generate(
                    input_ids=input_ids,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    top_k=cfg.top_k,
                    do_sample=True,
                    eos_token_id=tokenizer.eos_token_id,
                ) # 用策略模型生成回答
                attn_mask = torch.ones_like(full_ids, dtype=torch.long)

                # 计算旧策略的对数概率
                old_logp = scalar_logprob_of_suffix(policy.base, full_ids, prompt_len=prompt_len)
                # 计算参考模型的对数概率（用于 KL 散度）
                ref_logp = scalar_logprob_of_suffix(ref, full_ids, prompt_len=prompt_len)
                # 奖励模型打分
                reward = float(rm(full_ids, attn_mask).detach().cpu().item())
                # 减去 KL 散度惩罚
                reward = reward - cfg.kl_beta * float(old_logp - ref_logp) # 计算分数
                # 价值网络预测
                _, v = policy(full_ids, attn_mask)
                v0 = float(v.detach().cpu().item())
                # 计算优势函数 最终奖励 = 奖励模型打分 - β × KL(策略模型 || 参考模型)
                adv = float(reward - v0) # 实际奖励 - 预期奖励 = 优势 大于0说明这个回答比预期的好，增加其概率，小于0降低概率

            trajectories.append(
                {
                    "prompt_len": prompt_len,
                    "input_ids": full_ids,
                    "attention_mask": torch.ones_like(full_ids, dtype=torch.long),
                    "old_logp": float(old_logp),
                    "reward": float(reward),
                    "adv": float(adv),
                }
            )

        # Actor-Critic架构
        # PPO update (sequence-level) PPO 更新阶段
        for _ in range(int(cfg.ppo_epochs)):
            random.shuffle(trajectories)
            for tr in trajectories:
                input_ids = tr["input_ids"]
                attention_mask = tr["attention_mask"]
                old_logp = torch.tensor(tr["old_logp"], device=device, dtype=torch.float32)
                reward = torch.tensor(tr["reward"], device=device, dtype=torch.float32)
                adv = torch.tensor(tr["adv"], device=device, dtype=torch.float32)

                new_logp = logprob_of_suffix(policy.base, input_ids, prompt_len=tr["prompt_len"])
                ratio = torch.exp(new_logp - old_logp)
                # PPO-Clip 算法（核心）防止策略更新过大导致训练崩溃
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) # PPO-Clip 算法
                policy_loss = -torch.min(ratio * adv, clipped * adv) # ⚠️

                _, v_pred = policy(input_ids, attention_mask)
                value_loss = F.mse_loss(v_pred, reward) # ⚠️

                loss = policy_loss + cfg.vf_coef * value_loss

                optimizer.zero_grad(set_to_none=True)
                accelerator.backward(loss)
                optimizer.step()

        if is_main:
            dt = time.time() - t0
            avg_r = sum(x["reward"] for x in trajectories) / max(1, len(trajectories))
            accelerator.print(f"[ppo] update={update+1}/{cfg.updates} avg_reward={avg_r:.4f} time={dt:.1f}s")

    if is_main:
        save_dir = cfg.output_dir
        accelerator.print(f"[ppo] saving to {save_dir}")
        policy.save_pretrained(save_dir, cfg=PpoPolicyConfig(base_model_dir=policy_cfg.base_model_dir, pooling=policy_cfg.pooling))
        tokenizer.save_pretrained(save_dir)
        accelerator.print("[ppo] done")


if __name__ == "__main__":
    train_ppo()
