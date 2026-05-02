import tiktoken
import json

# 把 GPT2 模型的完整词表（50257 个 token）导出成一个 JSON 文件

tokenizer = tiktoken.get_encoding("gpt2")

# 创建词表字典
vocab_dict = {}
for token_id in range(tokenizer.n_vocab):
    token_text = tokenizer.decode([token_id])
    vocab_dict[token_id] = token_text

# 保存到 JSON 文件
with open('gpt2_vocab.json', 'w', encoding='utf-8') as f:
    json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

print(f"词表已保存到 gpt2_vocab.json，共 {tokenizer.n_vocab} 个 tokens")
