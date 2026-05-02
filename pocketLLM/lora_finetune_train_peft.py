import sys
import json
from dataclasses import asdict
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd


def _require_transformers():
    try:
        import torch
        from transformers import (  # type: ignore
            AutoModelForCausalLM,
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "未安装依赖：需要 `transformers`（以及训练时常用的 accelerate）。\n"
            "可选安装示例：pip install transformers accelerate"
        ) from exc
    return (
        torch,
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )


def _instruction_to_text(entry):
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry.get("input") else ""
    response_text = f"\n\n### Response:\n{entry['output']}"
    return instruction_text + input_text + response_text


class _SpamClsDataset:
    def __init__(self, df, tokenizer, max_length: int):
        self.labels = df["Label"].tolist()
        self.texts = df["Text"].tolist()
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        item["labels"] = int(self.labels[idx])
        return item


class _InstructionCausalDataset:
    def __init__(self, data, tokenizer, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = _instruction_to_text(self.data[idx])
        item = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        return item


def _prepare_spam_dfs(data_path: str):
    df = pd.read_csv(data_path, sep="\t", header=None, names=["Label", "Text"])
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    df["Label"] = df["Label"].map({"ham": 0, "spam": 1})
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    split_idx = int(len(df) * 0.9)
    train_df = df[:split_idx]
    eval_df = df[split_idx:]
    return train_df, eval_df


def _set_padding(tokenizer, model):
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id


def _compute_accuracy(eval_pred):
    import numpy as np

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float((preds == labels).mean())}


if __name__ == "__main__":
    """
    基于第三方库（Transformers + PEFT）的 LoRA 微调训练示例。

    特点：
      - 依赖 HuggingFace Transformers 的模型/Trainer
      - 依赖 PEFT 的 LoRA 实现（不走 utils/lora.py 的纯 PyTorch 注入）
      - 训练结束仅保存 adapter（以及 tokenizer），不保存整模

    说明：
      - 如在离线环境，请将 --model_name_or_path 指向本地已缓存/下载的模型目录
      - 该示例默认对 GPT-2 族模型生效（target_modules 默认是 c_attn/c_proj）
    """

    parser = ArgumentParser()
    parser.add_argument("--task", type=str, choices=["classification", "instruction"], required=True)
    parser.add_argument("--data_path", type=str, required=True)

    parser.add_argument("--model_name_or_path", type=str, default="gpt2")
    parser.add_argument("--adapter_dir", type=str, default="outputs/lora_peft/adapter")
    parser.add_argument("--output_dir", type=str, default="outputs/lora_peft/runs")

    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="c_attn,c_proj")

    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--do_eval", action="store_true")
    args = parser.parse_args()

    if not Path(args.data_path).exists():
        print(f"用于微调的原始数据文件 {args.data_path} 不存在")
        sys.exit(2)

    (
        torch,
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    ) = _require_transformers()

    from utils.lora_peft import PeftLoraArgs, inject_lora, print_trainable_parameters

    torch.manual_seed(int(args.seed))

    target_modules = [s.strip() for s in args.target_modules.split(",") if s.strip()]
    if len(target_modules) == 0:
        raise ValueError("--target_modules 不能为空（例如：c_attn,c_proj）")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    if args.task == "classification":
        base_model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            num_labels=2,
        )
        _set_padding(tokenizer, base_model)

        lora_args = PeftLoraArgs(
            task_type="seq_cls",
            r=args.rank,
            alpha=args.alpha,
            dropout=args.lora_dropout,
            target_modules=tuple(target_modules),
            modules_to_save=("score",),
        )
        model = inject_lora(base_model, lora_args)

        train_df, eval_df = _prepare_spam_dfs(args.data_path)
        train_dataset = _SpamClsDataset(train_df, tokenizer=tokenizer, max_length=args.max_length)
        eval_dataset = _SpamClsDataset(eval_df, tokenizer=tokenizer, max_length=args.max_length)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

        compute_metrics = _compute_accuracy if args.do_eval else None
    else:
        base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
        _set_padding(tokenizer, base_model)

        lora_args = PeftLoraArgs(
            task_type="causal_lm",
            r=args.rank,
            alpha=args.alpha,
            dropout=args.lora_dropout,
            target_modules=tuple(target_modules),
        )
        model = inject_lora(base_model, lora_args)

        with open(args.data_path) as f:
            data = json.load(f)

        split_idx = int(len(data) * 0.9)
        train_data = data[:split_idx]
        eval_data = data[split_idx:]
        train_dataset = _InstructionCausalDataset(train_data, tokenizer=tokenizer, max_length=args.max_length)
        eval_dataset = _InstructionCausalDataset(eval_data, tokenizer=tokenizer, max_length=args.max_length)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        compute_metrics = None

    print_trainable_parameters(model)

    eval_strategy = "steps" if args.do_eval else "no"
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        evaluation_strategy=eval_strategy,
        eval_steps=max(1, args.logging_steps),
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    adapter_dir = Path(args.adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    meta = {
        "task": args.task,
        "model_name_or_path": args.model_name_or_path,
        "lora": asdict(lora_args),
        "max_length": int(args.max_length),
    }
    with open(adapter_dir / "adapter_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"PEFT LoRA adapter 已保存到 {adapter_dir}")
