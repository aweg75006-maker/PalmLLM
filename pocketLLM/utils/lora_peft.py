"""
基于第三方库（PEFT）的 LoRA 注入/保存/加载工具。

本文件是对 utils/lora.py（纯 PyTorch LoRA 实现）的替代演示版本：
  - 依赖 `peft` 与 `transformers`
  - 不修改原有模型代码，只在训练脚本中对 HF 模型进行 LoRA 包装
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union


def _require_peft():
    try:
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "未安装依赖：需要 `peft` 与 `transformers`。\n"
            "可选安装示例：pip install peft transformers accelerate"
        ) from exc
    return LoraConfig, PeftModel, TaskType, get_peft_model


@dataclass(frozen=True)
class PeftLoraArgs:
    task_type: str = "causal_lm"  # causal_lm | seq_cls | token_cls
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: Sequence[str] = ("c_attn", "c_proj")
    bias: str = "none"  # none | all | lora_only
    modules_to_save: Optional[Sequence[str]] = None


def build_lora_config(args: PeftLoraArgs):
    LoraConfig, _, TaskType, _ = _require_peft()

    task_map = {
        "causal_lm": TaskType.CAUSAL_LM,
        "seq_cls": TaskType.SEQ_CLS,
        "token_cls": TaskType.TOKEN_CLS,
    }
    if args.task_type not in task_map:
        raise ValueError(f"Unsupported task_type={args.task_type!r}, expected one of {sorted(task_map.keys())}")

    return LoraConfig(
        task_type=task_map[args.task_type],
        r=int(args.r),
        lora_alpha=int(args.alpha),
        lora_dropout=float(args.dropout),
        target_modules=list(args.target_modules),
        bias=str(args.bias),
        modules_to_save=list(args.modules_to_save) if args.modules_to_save else None,
    )


def inject_lora(model, args: PeftLoraArgs):
    """
    使用 PEFT 将 LoRA adapter 注入到 HuggingFace 模型中。

    adapter 指 LoRA 微调新增的小矩阵，非大模型权重

    返回值是一个包装后的 PeftModel / LoRA model（可直接用于训练与 save_pretrained）。
    """
    _, _, _, get_peft_model = _require_peft()
    config = build_lora_config(args)
    return get_peft_model(model, config)


def print_trainable_parameters(model) -> str:
    trainable = 0
    total = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    pct = 0.0 if total == 0 else (100.0 * trainable / total)
    msg = f"Trainable params: {trainable:,} | All params: {total:,} | Trainable%: {pct:.2f}%"
    print(msg)
    return msg


def save_lora_adapter(model, adapter_dir: Union[str, Path]):
    """
    保存 LoRA adapter（不保存底座模型）。
    """
    adapter_dir = Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))


def load_lora_adapter(base_model, adapter_dir: Union[str, Path], *, is_trainable: bool = False):
    """
    将 adapter 加载到 base_model 上，返回 PeftModel。
    """
    _, PeftModel, _, _ = _require_peft()
    adapter_dir = Path(adapter_dir)
    return PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=is_trainable)


def merge_lora_and_unload(model):
    """
    将 LoRA 权重合并回底座权重并卸载 adapter（得到一个普通 HF 模型）。
    """
    if not hasattr(model, "merge_and_unload"):
        raise TypeError("当前 model 不支持 merge_and_unload（可能不是 PEFT LoRA 模型）")
    return model.merge_and_unload()
