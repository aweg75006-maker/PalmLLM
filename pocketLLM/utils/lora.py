import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
        直接替换 nn.Linear 的、自带 LoRA 适配器的线性层。散落在模型各个角落，无固定路径

        前向传播公式：
        y = xW^T + b + scale * ( dropout(x)  A^T B^T )

        其中：
        A：形状为 (r, in_features) 的低秩矩阵
        B：形状为 (out_features, r) 的低秩矩阵
    """

    def __init__(self, in_features, out_features, r=8, lora_alpha=16, lora_dropout=0.0, bias=True):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout and lora_dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)

        self.reset_parameters()

    @classmethod
    def from_linear(cls, linear, r=8, lora_alpha=16, lora_dropout=0.0):
        if not isinstance(linear, nn.Linear):
            raise TypeError("from_linear expects an nn.Linear")
        new = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=linear.bias is not None,
        )
        with torch.no_grad():
            new.weight.copy_(linear.weight)
            if linear.bias is not None:
                new.bias.copy_(linear.bias)
        return new

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)

        nn.init.zeros_(self.lora_B.weight)
        nn.init.normal_(self.lora_A.weight, std=0.02)

    def forward(self, x):
        result = torch.nn.functional.linear(x, self.weight, self.bias)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return result + lora_out


def _set_module_by_name(model, module_name, new_module):
    parts = module_name.split(".")
    parent = model
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def inject_lora(
    model,
    r=8,
    lora_alpha=16,
    lora_dropout=0.0,
    target_modules=("W_query", "W_key", "W_value", "out_proj", "ffn.layers.0", "ffn.layers.2"),
):
    """
    将选中的 nn.Linear 模块替换为 LoRALinear 模块。

    target_modules 按后缀名称进行匹配：
      - 精确匹配属性名称，例如 "W_query"
      - 或按点分隔的后缀匹配，例如 "ffn.layers.0"
   """
    replaced = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if any(name.endswith(suffix) for suffix in target_modules):
            new_module = LoRALinear.from_linear(module, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            _set_module_by_name(model, name, new_module)
            replaced.append(name)
    return replaced


def mark_only_lora_as_trainable(model, train_bias=False):
    """
    注意 这个是 手写 LoRA 模块（LoRALinear）+ 手动冻结/解冻参数 所以必须显式写 mark_only_lora_as_trainable() 来控制 requires_grad

    我新增的（utils/lora_peft.py）这套是用 PEFT 来做 LoRA：
    inject_lora() 里调用 peft.get_peft_model(model, LoraConfig(...)) 后，
    PEFT 会把 LoRA adapter 挂到模型里，并把“可训练参数”设置成只训练 LoRA（以及你指定的 modules_to_save，
    我在分类任务里加了 ("score",) 以便分类头也能训练/保存）。
    所以不需要你再写一段“遍历所有参数全部冻结，再挑 LoRA A/B 解冻”的代码；
    这件事由 PEFT 内部完成了。

    """
    for param in model.parameters():
        param.requires_grad = False

    for module in model.modules():
        if isinstance(module, LoRALinear):
            # 只解冻 LoRA 的小参数
            module.lora_A.weight.requires_grad = True
            module.lora_B.weight.requires_grad = True
            if train_bias and module.bias is not None:
                module.bias.requires_grad = True


def lora_state_dict(model):
    """
    Return a minimal state dict containing only LoRA adapter weights.
    Keys are the full module path to keep loading straightforward.
    """
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{name}.lora_A.weight"] = module.lora_A.weight.detach().cpu()
            state[f"{name}.lora_B.weight"] = module.lora_B.weight.detach().cpu()
    return state


def load_lora_state_dict(model, state_dict, strict=True):
    missing = []
    unexpected = []
    for key, value in state_dict.items():
        if not key.endswith(".weight"):
            unexpected.append(key)
            continue
        module_path, param_name = key.rsplit(".", 1)
        if not (module_path.endswith(".lora_A") or module_path.endswith(".lora_B")):
            unexpected.append(key)
            continue
        owner_path, lora_part = module_path.rsplit(".", 1)
        owner = dict(model.named_modules()).get(owner_path)
        if owner is None or not isinstance(owner, LoRALinear):
            missing.append(key)
            continue
        target = owner.lora_A if lora_part == "lora_A" else owner.lora_B
        if target.weight.shape != value.shape:
            raise ValueError(f"Shape mismatch for {key}: expected {target.weight.shape}, got {value.shape}")
        with torch.no_grad():
            target.weight.copy_(value.to(target.weight.device, dtype=target.weight.dtype))

    if strict and (missing or unexpected):
        msg = []
        if missing:
            msg.append(f"Missing keys: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        if unexpected:
            msg.append(f"Unexpected keys: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
        raise RuntimeError("LoRA state dict load failed. " + " ".join(msg))

    return {"missing_keys": missing, "unexpected_keys": unexpected}

