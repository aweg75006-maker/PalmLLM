import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoeDecoderConfig:
    # 词汇表大小
    vocab_size: int
    # 最大序列长度
    max_seq_len: int = 1024

    # 模型维度
    d_model: int = 512
    # 解码器层数
    n_layers: int = 12
    # 注意力头数
    n_heads: int = 8

    # MoE 前馈网络配置
    # 专家数量
    n_experts: int = 8
    # 每个token选择的专家数量
    top_k: int = 2
    # 专家内部维度
    d_ff: int = 2048
    # 专家容量系数
    capacity_factor: float = 1.25

    # dropout概率
    dropout: float = 0.0
    # 层归一化的epsilon
    layer_norm_eps: float = 1e-5

    # 注意力内存优化
    # 注意力分块大小，0表示不分块
    attn_chunk_size: int = 0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        # 可学习的缩放参数
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算方差
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        # 归一化并缩放
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


def _causal_mask_for_chunk(q_start: int, q_len: int, kv_len: int, device: torch.device) -> torch.Tensor:
    # True表示允许注意力，我们会转换为加法掩码
    # 查询位置 [q,1]
    q_pos = torch.arange(q_start, q_start + q_len, device=device).unsqueeze(-1)
    # 键位置 [1,kv]
    k_pos = torch.arange(0, kv_len, device=device).unsqueeze(0)
    # 因果掩码：键位置 <= 查询位置
    allow = k_pos <= q_pos  # [q,kv]
    # 转换为加法掩码：0表示允许，-inf表示屏蔽
    mask = torch.zeros((q_len, kv_len), device=device, dtype=torch.float32)
    mask = mask.masked_fill(~allow, float("-inf"))
    return mask


class SelfAttention(nn.Module):
    def __init__(self, config: MoeDecoderConfig) -> None:
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        # QKV投影层
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        # 输出投影层
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [批次大小, 序列长度, 模型维度]
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)  # [b,s,3d]
        q, k, v = qkv.chunk(3, dim=-1)

        # 调整形状为多头注意力格式 [b,h,s,hd]
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)

        # 注意力分块处理
        chunk = int(self.config.attn_chunk_size) if self.config.attn_chunk_size else 0
        if chunk <= 0 or chunk >= seqlen:
            # 使用PyTorch的缩放点积注意力（自动选择最优内核）
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        else:
            outs = []
            for q_start in range(0, seqlen, chunk):
                q_end = min(q_start + chunk, seqlen)
                q_chunk = q[:, :, q_start:q_end, :]  # [b,h,q,hd]
                # 生成分块因果掩码
                mask = _causal_mask_for_chunk(q_start=q_start, q_len=q_end - q_start, kv_len=seqlen, device=x.device)
                attn_chunk = F.scaled_dot_product_attention(
                    q_chunk,
                    k,
                    v,
                    attn_mask=mask,
                    dropout_p=0.0,
                    is_causal=False,
                )
                outs.append(attn_chunk)
            # 拼接所有分块结果
            attn = torch.cat(outs, dim=2)  # [b,h,s,hd]

        # 恢复形状并投影输出
        attn = attn.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        attn = self.out(attn)
        return self.dropout(attn)


class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 将输入切分为两部分，使用silu门控激活
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class ExpertFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.act = SwiGLU()
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 专家前馈网络计算
        return self.dropout(self.w2(self.act(self.w1(x))))


class MoeFFN(nn.Module):
    def __init__(self, config: MoeDecoderConfig) -> None:
        super().__init__()
        self.config = config
        # 路由层（门控网络）
        self.router = nn.Linear(config.d_model, config.n_experts, bias=False)
        # 专家列表
        self.experts = nn.ModuleList(
            [ExpertFFN(config.d_model, config.d_ff, config.dropout) for _ in range(config.n_experts)]
        )

    def compute_load_balancing_loss(self, router_probs: torch.Tensor, token_expert_mask: torch.Tensor) -> torch.Tensor:
        """
        计算负载均衡辅助损失，鼓励router均匀分配token到各个专家
        
        Args:
            router_probs: [t, e] 每个token分配给各专家的概率
            token_expert_mask: [t, e] 每个token实际分配的专家(one-hot或multi-hot)
        
        Returns:
            标量损失值
        """
        num_experts = self.config.n_experts
        
        # 计算每个专家接收到的token比例
        expert_usage = token_expert_mask.sum(dim=0)  # [e]
        expert_usage_frac = expert_usage / (expert_usage.sum() + 1e-8)
        
        # 计算router概率的均值
        mean_router_probs = router_probs.mean(dim=0)  # [e]
        
        # 负载均衡损失：鼓励均匀分布
        # 理想情况下，每个专家应该接收 1/num_experts 的token
        uniform_dist = 1.0 / num_experts
        load_balancing_loss = ((expert_usage_frac - uniform_dist) ** 2).sum()
        
        # 也可以加上router概率的熵正则化，鼓励不确定性
        entropy = -(router_probs * (router_probs + 1e-8).log()).sum(dim=-1).mean()
        entropy_loss = -entropy  # 最大化熵
        
        return load_balancing_loss + 0.01 * entropy_loss

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # x: [b,s,d] -> 展平为token序列 [t,d]
        bsz, seqlen, d_model = x.shape
        tokens = x.view(bsz * seqlen, d_model)

        # 路由预测：每个token分配给专家的概率
        logits = self.router(tokens)  # [t,e]
        probs = F.softmax(logits, dim=-1, dtype=torch.float32).to(tokens.dtype)

        # 选择top-k专家
        top_k = max(1, int(self.config.top_k))
        top_k = min(top_k, self.config.n_experts)
        topv, topi = torch.topk(probs, k=top_k, dim=-1)  # [t,k]

        # 构建token-expert分配掩码（用于辅助损失）
        token_expert_mask = torch.zeros_like(probs)
        token_expert_mask.scatter_(1, topi, 1.0)  # [t,e]

        # 专家容量限制（防止单个专家负载过高）
        t = tokens.shape[0]
        cap = int(math.ceil(self.config.capacity_factor * t / self.config.n_experts))
        cap = max(1, cap)

        out = torch.zeros_like(tokens)
        for expert_id in range(self.config.n_experts):
            # 筛选路由到当前专家的token
            routed = (topi == expert_id)  # [t,k]
            if not routed.any():
                continue

            token_idx = routed.any(dim=-1).nonzero(as_tuple=False).squeeze(-1)  # [n]
            # 超过容量则截断
            if token_idx.numel() > cap:
                token_idx = token_idx[:cap]

            # 选取对应token并通过专家网络
            x_e = tokens.index_select(0, token_idx)  # [n,d]
            y_e = self.experts[expert_id](x_e)  # [n,d]

            # 计算门控权重并加权输出
            routed_e = routed.index_select(0, token_idx)  # [n,k]
            gate_e = (topv.index_select(0, token_idx) * routed_e.to(topv.dtype)).sum(dim=-1)  # [n]
            out.index_add_(0, token_idx, y_e * gate_e.unsqueeze(-1))

        # 计算辅助损失
        aux_loss = self.compute_load_balancing_loss(probs, token_expert_mask)

        # 恢复原始形状
        return out.view(bsz, seqlen, d_model), aux_loss


class MoeDecoderLayer(nn.Module):
    def __init__(self, config: MoeDecoderConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.d_model, eps=config.layer_norm_eps)
        self.attn = SelfAttention(config)
        self.norm2 = RMSNorm(config.d_model, eps=config.layer_norm_eps)
        self.moe = MoeFFN(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 残差连接 + 自注意力
        x = x + self.dropout(self.attn(self.norm1(x)))
        # 残差连接 + MoE前馈网络
        moe_out, aux_loss = self.moe(self.norm2(x))
        x = x + self.dropout(moe_out)
        return x, aux_loss


class MoeDecoderLM(nn.Module):
    def __init__(self, config: MoeDecoderConfig) -> None:
        super().__init__()
        self.config = config

        # 词嵌入
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        # 位置嵌入
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        # 解码器层堆叠
        self.layers = nn.ModuleList([MoeDecoderLayer(config) for _ in range(config.n_layers)])
        self.norm_f = RMSNorm(config.d_model, eps=config.layer_norm_eps)
        # 语言模型头
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 权重绑定：词嵌入和lm_head共享权重
        self.lm_head.weight = self.tok_emb.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        return_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # input_ids: [批次大小, 序列长度]
        bsz, seqlen = input_ids.shape
        if seqlen > self.config.max_seq_len:
            raise ValueError(f"序列长度 {seqlen} 超过最大长度 {self.config.max_seq_len}")

        # 生成位置编码
        pos = torch.arange(seqlen, device=input_ids.device).unsqueeze(0).expand(bsz, seqlen)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)

        # 逐层传播，累积辅助损失
        total_aux_loss = 0.0
        for layer in self.layers:
            x, aux_loss = layer(x)
            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss

        x = self.norm_f(x)
        logits = self.lm_head(x)  # [b,s,v]
        
        # 平均所有层的辅助损失
        avg_aux_loss = total_aux_loss / len(self.layers) if len(self.layers) > 0 else None
        
        return logits, (x if return_hidden_states else None), avg_aux_loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        # 朴素的自回归生成（无KV缓存，适用于小型模型演示）
        out = input_ids
        for _ in range(max_new_tokens):
            logits, _, _ = self(out[:, -self.config.max_seq_len :])
            next_logits = logits[:, -1, :]

            # 温度系数调节
            if temperature and temperature != 1.0:
                next_logits = next_logits / float(temperature)

            if do_sample:
                # 采样生成
                if top_k and top_k > 0:
                    v, idx = torch.topk(next_logits, k=min(top_k, next_logits.shape[-1]), dim=-1)
                    probs = F.softmax(v, dim=-1)
                    next_token = idx.gather(-1, torch.multinomial(probs, num_samples=1))
                else:
                    probs = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
            else:
                # 贪心生成
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            # 拼接生成的token
            out = torch.cat([out, next_token], dim=1)
            if eos_token_id is not None:
                if (next_token == eos_token_id).all():
                    break

        return out

    def save_pretrained(self, save_dir: str) -> None:
        # 保存模型和配置
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, ensure_ascii=False, indent=2)
        torch.save(self.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))

    @staticmethod
    def from_pretrained(load_dir: str, map_location: str = "cpu") -> "MoeDecoderLM":
        # 从本地加载模型和配置
        with open(os.path.join(load_dir, "config.json"), "r", encoding="utf-8") as f:
            cfg = MoeDecoderConfig(**json.load(f))
        model = MoeDecoderLM(cfg)
        sd = torch.load(os.path.join(load_dir, "pytorch_model.bin"), map_location=map_location)
        model.load_state_dict(sd, strict=True)
        return model
