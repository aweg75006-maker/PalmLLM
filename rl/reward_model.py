import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn

from model.moe_decoder_lm import MoeDecoderConfig, MoeDecoderLM


@dataclass
class RewardModelConfig:
    base_model_dir: str
    pooling: str = "mean"  # "mean" | "last"


class RewardModel(nn.Module):
    """
    A minimal Reward Model (RM) for preference learning:
      score(prompt, answer) -> scalar reward
    """

    def __init__(self, base: MoeDecoderLM, pooling: str = "mean") -> None:
        super().__init__()
        self.base = base
        self.pooling = pooling
        self.value_head = nn.Linear(base.config.d_model, 1, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        _, hidden, _ = self.base(input_ids, return_hidden_states=True)
        if hidden is None:
            raise RuntimeError("base model did not return hidden states")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        if self.pooling == "last":
            # last non-pad token
            idx = attention_mask.long().sum(dim=1).clamp_min(1) - 1  # [b]
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), idx]  # [b,d]
        else:
            mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        score = self.value_head(pooled).squeeze(-1)  # [b]
        return score

    def save_pretrained(self, save_dir: str, rm_cfg: RewardModelConfig) -> None:
        os.makedirs(save_dir, exist_ok=True)
        self.base.save_pretrained(os.path.join(save_dir, "base"))
        torch.save(self.value_head.state_dict(), os.path.join(save_dir, "reward_head.bin"))
        with open(os.path.join(save_dir, "rm_config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(rm_cfg), f, ensure_ascii=False, indent=2)

    @staticmethod
    def from_pretrained(load_dir: str, map_location: str = "cpu") -> "RewardModel":
        with open(os.path.join(load_dir, "rm_config.json"), "r", encoding="utf-8") as f:
            rm_cfg = RewardModelConfig(**json.load(f))

        base = MoeDecoderLM.from_pretrained(os.path.join(load_dir, "base"), map_location=map_location)
        rm = RewardModel(base=base, pooling=rm_cfg.pooling)
        sd = torch.load(os.path.join(load_dir, "reward_head.bin"), map_location=map_location)
        rm.value_head.load_state_dict(sd, strict=True)
        return rm


def init_reward_model_from_base(base_dir: str, pooling: str = "mean", map_location: str = "cpu") -> tuple[RewardModel, RewardModelConfig]:
    base = MoeDecoderLM.from_pretrained(base_dir, map_location=map_location)
    rm = RewardModel(base=base, pooling=pooling)
    return rm, RewardModelConfig(base_model_dir=base_dir, pooling=pooling)
