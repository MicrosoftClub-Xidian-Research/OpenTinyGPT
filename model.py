import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import config as config

'''
X's shape is (B, T, C)
B:Batch Size 批次大小
T:Time/Sequence Length 输入的文本含有token的数量
C:Channel/Embedding Dimension 特征维度 每个token对应的特征维度

'''
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert config.n_embed % config.n_head == 0

        # input -> query, key, value
        self.c_attn = nn.Linear(config.n_embed, 3*config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        Batch, Time, Channel = x.size()

        # 1. projection & split
        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)

        # 2. transfer to multi-head
        k = k.view(Batch,Time,self.n_head, Channel // self.n_head).transpose(1, 2)
        q = q.view(Batch,Time,self.n_head, Channel // self.n_head).transpose(1, 2)
        v = v.view(Batch,Time,self.n_head, Channel // self.n_head).transpose(1, 2)

        # 3. Attention Scores
        # 实际上这一步，我们需要计算词与词之间的相互的关注程度
        att = (q @ k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1)))

        # 4. Causal Masking
        att = att.masked_fill(self.bias[:,:,:Time,:Time] == 0, float('-inf'))

        # 5. Softmax
        att = F.softmax(att,dim=-1)
        att = self.attn_dropout(att)

        # 6. Weighted Sum
        y = att @ v

        # 7. Re-assemble
        y = y.transpose(1,2).contiguous().view(Batch, Time, Channel)

        return self.resid_dropout(self.c_proj(y))

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4*config.n_embed)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4*config.n_embed, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        
        return x

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(vocab_size, config.n_embed)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embed)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block() for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embed)
        self.lm_head = nn.Linear(config.n_embed, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:

        device = idx.device
        b, t = idx.size()

        assert t <= config.block_size, f"Cannot forward sequence of length {t}, block size is only {config.block_size}"
        pos = torch.arange(0, t ,dtype=torch.long, device=device)
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(pos)

        x = self.drop(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        
        for _ in range(max_new_tokens):
            # 1. 如果当前的句子太长了，超过了 block_size，就只截取最后 block_size 个字
            # 否则位置编码会越界报错
            idx_cond = idx if idx.size(1) <= config.block_size else idx[:, -config.block_size:]
            
            # 2. 前向传播，算出概率 (Logits)
            logits = self(idx_cond)
            
            # 3. 我们只关心最后一个时间步的预测（也就是根据所有上文，预测的下一个字）
            logits = logits[:, -1, :] # 形状变为 (Batch, vocab_size)
            
            # 4. 除以温度 (Temperature Scaling)
            # 温度越高，logits 差距越小，概率越平坦，生成越随机
            logits = logits / temperature
            
            # 5. Top-k 截断 (可选)
            # 把概率特别低的字直接把概率设为 -inf，不让它被选中
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # 6. Softmax 算出最终概率分布
            probs = F.softmax(logits, dim=-1)
            
            # 7. 根据概率采样 (Sampling)
            # torch.multinomial 就像掷骰子，概率大的面朝上的几率大
            idx_next = torch.multinomial(probs, num_samples=1) # (Batch, 1)
            
            # 8. 把新生成的字拼接到原来的序列后面
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx