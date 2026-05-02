import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention
from .layer_norm import LayerNorm

class ExpertFFN(nn.Module):
    """
    单个专家，本质上就是一个 FFN
    """
    def __init__(self, emb_dim, hidden_dim, drop_rate):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, emb_dim)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MoEFeedForward(nn.Module):
    """
    教学版 MoE-FFN：
    - router 决定每个 token 交给哪些 expert
    - top-k 路由
    - 返回 (输出, aux_loss)
    """
    def __init__(self, emb_dim, hidden_dim, num_experts=4, top_k=2, drop_rate=0.1):
        super().__init__()
        assert top_k <= num_experts, "top_k 必须小于等于 num_experts"

        self.emb_dim = emb_dim
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = nn.Linear(emb_dim, num_experts)
        self.experts = nn.ModuleList([
            ExpertFFN(emb_dim, hidden_dim, drop_rate)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        """
        x: [B, T, C]
        return:
            y: [B, T, C]
            aux_loss: 路由负载均衡损失
        """
        orig_shape = x.shape
        b, t, c = orig_shape
        x_flat = x.reshape(-1, c)  # [N, C], N = B*T

        # 1) Router logits / probs
        router_logits = self.router(x_flat)                    # [N, E]
        router_probs = F.softmax(router_logits, dim=-1)        # [N, E]

        # 2) Top-k 选择
        topk_probs, topk_idx = torch.topk(router_probs, self.top_k, dim=-1)  # [N, K], [N, K]

        # 归一化 top-k 权重
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # 3) 计算输出
        out = torch.zeros_like(x_flat)  # [N, C]

        # 简单实现：逐 expert 聚合
        for expert_id, expert in enumerate(self.experts):
            # 哪些 token 被分给了这个 expert
            mask = (topk_idx == expert_id)  # [N, K]
            token_mask = mask.any(dim=-1)    # [N]

            if token_mask.sum().item() == 0:
                continue

            selected_x = x_flat[token_mask]                 # [M, C]
            selected_out = expert(selected_x)               # [M, C]

            # 每个 token 对这个 expert 的权重
            selected_weights = topk_probs[token_mask] * mask[token_mask].float()  # [M, K]
            selected_weights = selected_weights.sum(dim=-1, keepdim=True)         # [M, 1]

            out[token_mask] += selected_out * selected_weights

        # 4) 负载均衡辅助损失
        # 目标：避免所有 token 都挤到少数 expert 上
        # 一个简单可用的近似版
        expert_usage = torch.zeros(self.num_experts, device=x.device)
        for expert_id in range(self.num_experts):
            expert_usage[expert_id] = (topk_idx == expert_id).any(dim=-1).float().mean()

        expert_prob_mean = router_probs.mean(dim=0)  # [E]

        # 让“路由概率”和“实际使用率”尽量接近
        aux_loss = self.num_experts * torch.sum(expert_prob_mean * expert_usage)

        return out.reshape(orig_shape), aux_loss
