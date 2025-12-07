import json
import random
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

import OpenTinyGPT.config as config
from OpenTinyGPT.model import GPT

random.seed(config.seed)
torch.manual_seed(config.seed)

def load_text_corpus() -> str:
    if config.data_source == "local":
        corpus_path = Path("corpus.txt")
        if not corpus_path.exists():
            raise FileNotFoundError("未找到 corpus.txt，请在当前目录放置语料文件，或改用线上数据源。")
        return corpus_path.read_text(encoding="utf-8")

    # 线上数据源
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(
            "未安装 datasets 库，请先 `pip install datasets` 或使用 requirements.txt。"
        ) from e

    if config.data_source == "wikitext2":
        print("使用 WikiText-2（英文）作为语料...")
        ds_train = load_dataset("wikitext", config.wikitext_config, split="train")
        ds_val = load_dataset("wikitext", config.wikitext_config, split="validation")
        texts = [x["text"] for x in ds_train] + [x["text"] for x in ds_val]
        combined = "\n".join(t for t in texts if t)
        if config.online_max_text_chars:
            combined = combined[: config.online_max_text_chars]
        return combined

    raise ValueError(f"未知数据源：{config.data_source}")

def main():
    text = load_text_corpus()

    # 根据 token_level 构建词表（词级/字符级）
    if getattr(config, "token_level", "char") == "word":
        import re
        tokens = re.findall(r"\w+|[^\w\s]", text)
        vocab = sorted(set(tokens))
        vocab_size = len(vocab)
        stoi = {tok: i for i, tok in enumerate(vocab)}
        itos = {i: tok for i, tok in enumerate(vocab)}

        def encode(s: str):
            toks = re.findall(r"\w+|[^\w\s]", s)
            return [stoi[t] for t in toks if t in stoi]

        def decode(ids):
            return " ".join(itos[i] for i in ids)

        data = torch.tensor(encode(text), dtype=torch.long)
    else:
        # 构建字符级词表
        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}

        def encode(s: str):
            return [stoi[ch] for ch in s if ch in stoi]

        def decode(ids):
            return "".join(itos[i] for i in ids)

        data = torch.tensor(encode(text), dtype=torch.long)
    if len(data) < config.block_size + 2:
        raise ValueError("语料过短，无法组成有效训练样本。请增大在线抽样或使用更大语料。")

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(
        f"Tokenization={getattr(config, 'token_level', 'char')} | vocab_size={vocab_size} | data_len={len(data)}"
    )
    sample_tokens = [itos[i] for i in range(min(15, vocab_size))]
    print(f"sample_tokens={sample_tokens}")

    def get_batch(split: str):
        source = train_data if split == "train" else val_data
        ix = torch.randint(len(source) - config.block_size - 1, (config.batch_size,))
        x = torch.stack([source[i : i + config.block_size] for i in ix])
        y = torch.stack([source[i + 1 : i + 1 + config.block_size] for i in ix])
        x = x.to(config.device)
        y = y.to(config.device)
        return x, y

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = []
            for _ in range(config.eval_iters):
                xb, yb = get_batch(split)
                logits = model(xb)
                loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses)
        model.train()
        return out

    model = GPT(vocab_size=vocab_size).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    print(
        f"设备={config.device} | vocab_size={vocab_size} | "
        f"block_size={config.block_size} | batch_size={config.batch_size} | 数据源={config.data_source}"
    )

    for it in range(config.max_iters):
        if it % config.eval_interval == 0:
            losses = estimate_loss()
            print(
                f"iter {it:04d}: train_loss={losses['train']:.4f} | val_loss={losses['val']:.4f}"
            )
        xb, yb = get_batch("train")
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(
        {"model_state_dict": model.state_dict(), "vocab_size": vocab_size},
        "model.pt",
    )
    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump({"stoi": stoi, "itos": itos}, f, ensure_ascii=False, indent=2)

    prompt = "The month after January is "
    enc = encode(prompt)
    if len(enc) == 0:
        print("提示词不在词表，使用回退 token 开始生成。")
        fallback_id = None
        if getattr(config, "token_level", "char") == "word":
            fallback_id = stoi.get("the")
        else:
            fallback_id = stoi.get(" ")
        if fallback_id is None:
            fallback_id = 0
        idx = torch.tensor([[fallback_id]], dtype=torch.long, device=config.device)
    else:
        idx = torch.tensor([enc], dtype=torch.long, device=config.device)
    generated = model.generate(idx, max_new_tokens=200, temperature=0.8, top_k=50)
    print("示例生成：")
    print(decode(generated[0].tolist()))

if __name__ == "__main__":
    main()