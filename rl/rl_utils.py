import torch
import torch.nn.functional as F


def logprob_of_suffix(model, input_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """
    Differentiable log p(y | x) for a single sample.
    input_ids: [1, s]
    """
    logits, _, _ = model(input_ids)
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)  # [1,s-1,v]
    next_ids = input_ids[:, 1:]  # [1,s-1]
    start = max(0, int(prompt_len) - 1)
    gather = logp[:, start:, :].gather(-1, next_ids[:, start:].unsqueeze(-1)).squeeze(-1)  # [1,gen]
    return gather.sum()


@torch.no_grad()
def scalar_logprob_of_suffix(model, input_ids: torch.Tensor, prompt_len: int) -> float:
    return float(logprob_of_suffix(model, input_ids, prompt_len).detach().cpu().item())
