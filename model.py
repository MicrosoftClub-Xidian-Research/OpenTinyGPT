import torch
import torch.nn as nn
import OpenTinyGPT.config as config

class GPT(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(vocab_size, config.n_embed)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embed)

        ## defien your gpt-style model here

    def forward(self, idx: torch.Tensor) -> torch.Tensor:

        ## define your training causal forward function here

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        
        ## define your generation function here