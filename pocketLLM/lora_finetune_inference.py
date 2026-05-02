import sys
import json
from pathlib import Path
from argparse import ArgumentParser

import tiktoken
import torch

from model.language_model import LanguageModel
from utils.load_gpt2_weights import load_gpt2_weights_into_model
from utils.model_inference import generate_text, classify_review
from utils.lora import inject_lora, load_lora_state_dict


if __name__ == "__main__":
    """
    基于 LoRA adapter 的推理（需要 base GPT-2 权重 + adapter 权重）。
    """
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, choices=["classification", "instruction"], required=True)
    parser.add_argument("--config", type=str, default="configs/gpt2_config_355M.json")
    parser.add_argument("--adapter_path", type=str, default="lora_adapter.pth")
    parser.add_argument("--gpt2_model_path", type=str, default="pytorch_model.bin")
    args = parser.parse_args()

    if not Path(args.adapter_path).exists():
        print(f"LoRA adapter 文件 {args.adapter_path} 不存在，请先训练")
        sys.exit()
    if not Path(args.gpt2_model_path).exists():
        print(f"GPT-2 模型权重文件 {args.gpt2_model_path} 不存在")
        sys.exit()
    if not Path(args.config).exists():
        print(f"模型配置文件 {args.config} 不存在")
        sys.exit()

    with open(args.config) as f:
        cfg = json.load(f)

    adapter = torch.load(args.adapter_path, weights_only=False, map_location="cpu")
    if adapter.get("task") != args.task:
        print(f"adapter.task={adapter.get('task')} 与 --task={args.task} 不匹配")
        sys.exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")

    model = LanguageModel(cfg).to(device)
    load_gpt2_weights_into_model(model, args.gpt2_model_path)

    lora_cfg = adapter.get("lora_config", {})
    inject_lora(
        model,
        r=int(lora_cfg.get("rank", 8)),
        lora_alpha=int(lora_cfg.get("alpha", 16)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.0)),
        target_modules=tuple(lora_cfg.get("target_modules", ["W_query", "W_key", "W_value", "out_proj", "ffn.layers.0", "ffn.layers.2"])),
    )
    load_lora_state_dict(model, adapter["lora"], strict=True)

    if args.task == "classification":
        model.out_head = torch.nn.Linear(cfg["emb_dim"], 2).to(device)
        model.out_head.load_state_dict(adapter["out_head"])

    model.eval()

    print("开始对话（输入'exit'退出）\n")
    if args.task == "classification":
        max_length = int(adapter.get("max_length", cfg["context_length"]))
        while True:
            input_text = input("用户: ")
            if input_text.lower() == "":
                print("输入不能为空！")
                continue
            if input_text.lower() == "exit":
                break
            label = classify_review(
                input_text=input_text,
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_length=max_length,
            )
            print(f"模型: {label}\n")
    else:
        while True:
            task_instruction_text = input("任务指令: ")
            if task_instruction_text.lower() == "":
                print("任务指令不能为空！")
                continue
            if task_instruction_text.lower() == "exit":
                break
            input_text = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request."
                f"\n\n### Instruction:\n{task_instruction_text}"
            )

            task_input_text = input("任务输入: ")
            if task_input_text.lower() == "exit":
                break
            input_text += f"\n\n### Input:\n{task_input_text}" if task_input_text else ""

            output_text = generate_text(
                input_text=input_text,
                model=model,
                tokenizer=tokenizer,
                context_length=cfg["context_length"],
                max_new_tokens=cfg["context_length"],
            )

            response_text = output_text[len(input_text) :].replace("### Response:", "").strip()
            print(f"模型: {response_text}\n")

