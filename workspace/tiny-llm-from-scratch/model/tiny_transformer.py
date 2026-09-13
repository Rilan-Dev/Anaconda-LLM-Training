import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer(
            "bias",
            mask.view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(C, dim=2)
        head_dim = C // self.n_head

        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.c_proj(y))


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)

        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.ffwd = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffwd(self.ln_2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)

        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            Block(config) for _ in range(config.n_layer)
        ])

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        if T > self.config.block_size:
            raise ValueError("Sequence length is greater than block_size")

        positions = torch.arange(0, T, device=idx.device)

        token_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(positions)

        x = self.dropout(token_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens=100,
        temperature=1.0,
        top_k=None,
        top_p=1.0,
        repetition_penalty=1.0,
        do_sample=True,
        eos_token_id=None,
        seed=None,
    ):
        """
        Generate text using different decoding strategies.
    
        Parameters
        ----------
        idx : Tensor
            Input token IDs of shape (B, T)
    
        max_new_tokens : int
            Maximum tokens to generate.
    
        temperature : float
            Controls randomness.
            Lower = more deterministic.
    
        top_k : int or None
            Keep only top-k tokens.
    
        top_p : float
            Nucleus sampling threshold.
    
        repetition_penalty : float
            Penalize previously generated tokens.
    
        do_sample : bool
            True  -> Sampling
            False -> Greedy decoding
    
        eos_token_id : int or None
            Stop generation when EOS is produced.
    
        seed : int or None
            Random seed for reproducibility.
        """
    
        self.eval()
    
        if seed is not None:
            torch.manual_seed(seed)
    
        for _ in range(max_new_tokens):
    
            idx_cond = idx[:, -self.config.block_size:]
    
            logits, _ = self(idx_cond)
    
            logits = logits[:, -1, :]
    
            # ----------------------------
            # Repetition Penalty
            # ----------------------------
            if repetition_penalty > 1.0:
    
                for batch in range(idx.size(0)):
    
                    used_tokens = torch.unique(idx[batch])
    
                    logits[batch, used_tokens] /= repetition_penalty
    
            # ----------------------------
            # Temperature
            # ----------------------------
            temperature = max(temperature, 1e-8)
    
            logits = logits / temperature
    
            # ----------------------------
            # Top-K
            # ----------------------------
            if top_k is not None:
    
                k = min(top_k, logits.size(-1))
    
                values, _ = torch.topk(logits, k)
    
                logits[logits < values[:, [-1]]] = -float("inf")
    
            # ----------------------------
            # Top-P (Nucleus Sampling)
            # ----------------------------
            if top_p < 1.0:
    
                sorted_logits, sorted_indices = torch.sort(
                    logits,
                    descending=True
                )
    
                sorted_probs = torch.softmax(
                    sorted_logits,
                    dim=-1
                )
    
                cumulative_probs = torch.cumsum(
                    sorted_probs,
                    dim=-1
                )
    
                remove = cumulative_probs > top_p
    
                remove[..., 1:] = remove[..., :-1].clone()
    
                remove[..., 0] = False
    
                sorted_logits[remove] = -float("inf")
    
                logits = torch.full_like(logits, -float("inf"))
    
                logits.scatter_(1, sorted_indices, sorted_logits)
    
            # ----------------------------
            # Greedy vs Sampling
            # ----------------------------
            if do_sample:
    
                probs = torch.softmax(logits, dim=-1)
    
                next_token = torch.multinomial(
                    probs,
                    num_samples=1
                )
    
            else:
    
                next_token = torch.argmax(
                    logits,
                    dim=-1,
                    keepdim=True
                )
    
            idx = torch.cat(
                (idx, next_token),
                dim=1
            )
    
            # ----------------------------
            # EOS
            # ----------------------------
            if eos_token_id is not None:
    
                if torch.all(next_token == eos_token_id):
    
                    break
    
        return idx


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
