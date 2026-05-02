import sys
import json
from pathlib import Path
from argparse import ArgumentParser
from functools import partial

import pandas as pd
import tiktoken
import torch
from torch.utils.data import DataLoader

from model.language_model import LanguageModel
from utils.dataset_loader import SpamDataset, InstructionDataset, custom_collate_fn
from utils.load_gpt2_weights import load_gpt2_weights_into_model
from utils.model_train import train_model
from utils.lora import inject_lora, mark_only_lora_as_trainable, lora_state_dict


def _prepare_spam_loaders(data_path, tokenizer, context_length, batch_size, num_workers):
    df = pd.read_csv(data_path, sep="\t", header=None, names=["Label", "Text"])
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    df["Label"] = df["Label"].map({"ham": 0, "spam": 1})
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    train_ratio = 0.9
    split_idx = int(len(df) * train_ratio)
    train_data = df[:split_idx]
    validate_data = df[split_idx:]

    max_len = max(len(tokenizer.encode(text)) for text in train_data["Text"])
    max_len = min(max_len, context_length)

    train_dataset = SpamDataset(train_data, tokenizer=tokenizer, max_length=max_len)
    validate_dataset = SpamDataset(validate_data, tokenizer=tokenizer, max_length=max_len)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    validate_loader = DataLoader(
        dataset=validate_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    return train_loader, validate_loader, max_len


def _prepare_instruction_loaders(data_path, tokenizer, context_length, batch_size, num_workers, device):
    with open(data_path) as f:
        data = json.load(f)

    train_ratio = 0.9
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    validate_data = data[split_idx:]

    customized_collate_fn = partial(custom_collate_fn, allowed_max_length=context_length, device=device)
    train_dataset = InstructionDataset(train_data, tokenizer=tokenizer)
    validate_dataset = InstructionDataset(validate_data, tokenizer=tokenizer)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    validate_loader = DataLoader(
        dataset=validate_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    return train_loader, validate_loader


if __name__ == "__main__":
    """
    基于 LoRA 的微调训练（不依赖 peft）。

    Args:
        --task: classification 或 instruction
        --config: 模型配置文件路径
        --data_path: 数据路径（分类 CSV / 指令 JSON）
        --adapter_path: 保存 LoRA adapter 权重路径（推荐保存 adapter 而非整模）
        --gpt2_model_path: GPT-2 权重（pytorch_model.bin）
    """
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, choices=["classification", "instruction"], required=True)
    parser.add_argument("--config", type=str, default="configs/gpt2_config_355M.json")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default="lora_adapter.pth")
    parser.add_argument("--gpt2_model_path", type=str, default="pytorch_model.bin")

    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    if not Path(args.data_path).exists():
        print(f"用于微调的原始数据文件 {args.data_path} 不存在")
        sys.exit()
    if not Path(args.gpt2_model_path).exists():
        print(f"用于微调的 GPT-2 模型权重文件 {args.gpt2_model_path} 不存在")
        sys.exit()
    if not Path(args.config).exists():
        print(f"模型配置文件 {args.config} 不存在")
        sys.exit()

    with open(args.config) as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")

    model = LanguageModel(cfg)
    model.to(device)
    load_gpt2_weights_into_model(model, args.gpt2_model_path)

    # 将 LoRA 模块注入到 model 中
    replaced = inject_lora(
        model,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.lora_dropout,
    )
    if len(replaced) == 0:
        raise RuntimeError("No target Linear modules were replaced by LoRA.")

    if args.task == "classification":
        model.out_head = torch.nn.Linear(cfg["emb_dim"], 2).to(device)

    # lora
    mark_only_lora_as_trainable(model)
    if args.task == "classification":
        for p in model.out_head.parameters():
            p.requires_grad = True

    if args.task == "classification":
        train_loader, validate_loader, max_len = _prepare_spam_loaders( # 不同的数据处理格式 spam
            data_path=args.data_path,
            tokenizer=tokenizer,
            context_length=cfg["context_length"],
            batch_size=args.batch_size,
            num_workers=0,
        )
        is_classification = True
        eval_freq, eval_iter = 50, 5
    else:
        train_loader, validate_loader = _prepare_instruction_loaders( # # 不同的数据处理格式 instruction
            data_path=args.data_path,
            tokenizer=tokenizer,
            context_length=cfg["context_length"],
            batch_size=args.batch_size,
            num_workers=0,
            device=device,
        )
        is_classification = False
        eval_freq, eval_iter = 5, 5

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),  # 生成器表达式
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_model(
        model,
        train_loader,
        validate_loader,
        optimizer,
        device,
        num_epochs=args.num_epochs,
        eval_freq=eval_freq,
        eval_iter=eval_iter,
        is_classification=is_classification,
    )

    payload = {
        "task": args.task,
        "config": cfg,
        "lora_config": {
            "rank": args.rank,
            "alpha": args.alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": ["W_query", "W_key", "W_value", "out_proj", "ffn.layers.0", "ffn.layers.2"],
        },
        "lora": lora_state_dict(model), # lora
    }
    if args.task == "classification":
        payload["out_head"] = model.out_head.state_dict()
        payload["max_length"] = int(max_len)

    torch.save(payload, args.adapter_path)
    print(f"LoRA adapter 已保存到 {args.adapter_path}")
