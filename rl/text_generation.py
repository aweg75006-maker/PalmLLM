import torch
import torch.nn.functional as F


@torch.no_grad()
def decode_generated_text(tokenizer, prompt_ids: torch.Tensor, full_ids: torch.Tensor) -> str:
    """
    prompt_ids: [1, p]
    full_ids:   [1, p+g]
    """
    gen_ids = full_ids[:, prompt_ids.shape[1] :]
    return tokenizer.decode(gen_ids[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)


@torch.no_grad()
def sequence_logprob(model, input_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """
    Returns log p(y | x) for the generated suffix y, where x length = prompt_len.
    input_ids: [b, s]
    """
    logits, _, _ = model(input_ids)
    # Predict token t+1 from position t
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)
    next_ids = input_ids[:, 1:]  # [b, s-1]

    # Generated tokens are positions >= prompt_len (in the original sequence),
    # which correspond to next_ids positions >= prompt_len-1.
    start = max(0, prompt_len - 1)
    gather = logp[:, start:, :].gather(-1, next_ids[:, start:].unsqueeze(-1)).squeeze(-1)  # [b, gen_len]
    return gather.sum(dim=-1)


@torch.no_grad()
def batched_generate(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: int = 0,
    do_sample: bool = True,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      input_ids: [b, p] (padded to max prompt len)
      full_ids:  [b, p+g] (variable g per sample is not supported; this uses greedy stop on EOS-all only)
    """
    enc = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    eos_id = tokenizer.eos_token_id
    full_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        do_sample=do_sample,
        eos_token_id=eos_id,
    )
    return input_ids, full_ids
