# coding=utf-8
import os
from dataclasses import dataclass
from typing import Iterable, Optional

import pyarrow.parquet as pq
import tokenizers
from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast


@dataclass
class TinyTokenizerConfig:
    parquet_file: str = "./data/my_train_dataset_3k.parquet"
    output_dir: str = "./model_save/tiny_tokenizer"
    vocab_size: int = 8000
    min_frequency: int = 2
    max_chars: int = 2_000_000  # simple safety cap for tiny corpora


def iter_text(table) -> Iterable[str]:
    cur = 0
    for p, r in zip(table["prompt"], table["response"]):
        txt = f"{p.as_py()}\n{r.as_py()}"
        cur += len(txt)
        yield txt
        if cur >= 2_000_000:
            break


def train_tiny_tokenizer(cfg: Optional[TinyTokenizerConfig] = None) -> None:
    cfg = cfg or TinyTokenizerConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)

    table = pq.read_table(cfg.parquet_file)

    special_tokens = ["[PAD]", "[EOS]", "[UNK]", "[BOS]", "[SEP]", "[CLS]", "[MASK]"]

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = tokenizers.normalizers.NFKC()
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel(add_prefix_space=False, use_regex=True)

    trainer = BpeTrainer(
        vocab_size=int(cfg.vocab_size),
        min_frequency=int(cfg.min_frequency),
        show_progress=True,
        special_tokens=special_tokens,
    )
    tokenizer.train_from_iterator(iter_text(table), trainer=trainer)

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    fast.save_pretrained(cfg.output_dir)
    print(f"[tiny_tokenizer] saved to {cfg.output_dir}")


if __name__ == "__main__":
    train_tiny_tokenizer()

