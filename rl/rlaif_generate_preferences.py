# coding=utf-8
import os
import sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import json
import random
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from transformers import PreTrainedTokenizerFast

from model.moe_decoder_lm import MoeDecoderLM
from rl.judges import HeuristicJudge, PairwiseJudge
from rl.text_generation import decode_generated_text


@dataclass
class RlaifConfig:
    model_dir: str = "./model_save/moe_pretrain"
    prompts_file: str = "./data/rlaif_prompts.jsonl"
    output_file: str = "./data/rlaif_preferences.jsonl"

    max_new_tokens: int = 128
    temperature: float = 0.9
    top_k: int = 50
    candidates_per_prompt: int = 2
    seed: int = 23333


def iter_prompts_jsonl(file: str) -> Iterable[str]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield obj["prompt"]


@torch.no_grad()
def generate_one(model: MoeDecoderLM, tokenizer: PreTrainedTokenizerFast, prompt: str, cfg: RlaifConfig, device: str) -> str:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    full = model.generate(
        input_ids=input_ids,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_k=cfg.top_k,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
    )
    return decode_generated_text(tokenizer, input_ids, full)


def rlaif_generate_preferences(cfg: Optional[RlaifConfig] = None) -> None:
    cfg = cfg or RlaifConfig()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.model_dir)
    model = MoeDecoderLM.from_pretrained(cfg.model_dir, map_location="cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    judge = HeuristicJudge()
    pick = PairwiseJudge()

    os.makedirs(os.path.dirname(cfg.output_file) or ".", exist_ok=True)
    out_cnt = 0
    with open(cfg.output_file, "w", encoding="utf-8") as w:
        for prompt in iter_prompts_jsonl(cfg.prompts_file):
            cands = []
            for i in range(int(cfg.candidates_per_prompt)):
                torch.manual_seed(cfg.seed + out_cnt * 13 + i)
                ans = generate_one(model, tokenizer, prompt, cfg, device=device)
                scored = judge.score(prompt, ans)
                cands.append((ans, scored))

            # pick best & worst as chosen/rejected
            cands.sort(key=lambda x: x[1].score, reverse=True)
            chosen, chosen_s = cands[0]
            rejected, rejected_s = cands[-1]

            obj = {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "chosen_score": chosen_s.score,
                "rejected_score": rejected_s.score,
                "judge": "heuristic",
                "judge_reason": {"chosen": chosen_s.reason, "rejected": rejected_s.reason},
            }
            w.write(json.dumps(obj, ensure_ascii=False) + "\n")
            out_cnt += 1

    print(f"[rlaif] wrote {out_cnt} pairs to {cfg.output_file}")


if __name__ == "__main__":
    rlaif_generate_preferences()
