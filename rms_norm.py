import typing
from typing import Optional, Callable, Iterable

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from jaxtyping import Float


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: Optional[float] = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones((d_model,)))

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x_squared = torch.mul(x, x)
        mean_x_squared = torch.sum(x_squared, -1, keepdim=True) / float(self.d_model)
        rms = torch.sqrt(mean_x_squared + self.eps)

        return self.weight * x / rms


class GELU(nn.Module):
    def forward(self, x: Float) -> Float:
        return x * 0.5 * (1 + torch.erf(x / 1.4142135623730951))


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gelu = GELU()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Float) -> Float:
        return self.w2(self.gelu(self.w1(x)))


class Softmax(nn.Module):
    def forward(self, x: Float) -> Float:
        x = x - torch.max(x, dim=-1, keepdim=True).values
        exp_x = torch.exp(x)
        return exp_x / torch.sum(exp_x, -1, keepdim=True)


class Attention(nn.Module):
    def __init__(self, attn_pdrop: float | None, residual_pdrop: float | None) -> None:
        super().__init__()
        self.attn_dropout = torch.nn.Dropout(attn_pdrop) if attn_pdrop is not None else torch.nn.Identity()
        self.residual_dropout = torch.nn.Dropout(residual_pdrop) if residual_pdrop is not None else torch.nn.Identity()

    def forward(
            self,
            keys: torch.FloatTensor,
            queries: torch.FloatTensor,
            values: torch.FloatTensor,
            mask: Optional[torch.BoolTensor]):
        d_k = keys.shape[-1]
        logits = torch.matmul(queries, keys.transpose(-2, -1)) / (d_k ** 0.5)
        masked_logits = torch.where(mask, -torch.inf, logits)
        normalized_logits = F.softmax(masked_logits, dim=-1)
        normalized_logits = self.attn_dropout(normalized_logits)
        return self.residual_dropout(torch.matmul(normalized_logits, values))


class MultiHeadSelfAttention(nn.Module):
    def __init__(
            self,
            num_heads: int,
            d_model: int,
            d_key: int,
            d_value: int,
            d_query: int,
            attn_pdrop: float | None,
            residual_pdrop: float | None) -> None:
        """
        d_key / value / query key / value / query projection size
        seq_len max sequence size
        """

        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_key = d_key
        self.d_query = d_query
        self.d_value = d_value
        self.k_proj = torch.nn.Linear(d_model, num_heads * d_key, bias=False)
        self.q_proj = torch.nn.Linear(d_model, num_heads * d_query, bias=False)
        self.v_proj = torch.nn.Linear(d_model, num_heads * d_value, bias=False)
        self.output_proj = torch.nn.Linear(num_heads * d_value, d_model, bias=False)
        self.attention = Attention(attn_pdrop, residual_pdrop)


    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        batch, seq_len, _ = x.shape
        
        # Project and reshape to (batch, seq_len, num_heads, d_*)
        queries = self.q_proj(x).view(batch, seq_len, self.num_heads, self.d_query)
        keys = self.k_proj(x).view(batch, seq_len, self.num_heads, self.d_key)
        values = self.v_proj(x).view(batch, seq_len, self.num_heads, self.d_value)
        
        # Transpose to (batch, num_heads, seq_len, d_*)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # Create causal mask
        mask = torch.triu(
            torch.ones((batch, self.num_heads, seq_len, seq_len), dtype=torch.bool, device=x.device), diagonal=1)

        # Apply attention
        attention = self.attention(keys, queries, values, mask)
        
        # Reshape back to (batch, seq_len, num_heads * d_value)
        attention = attention.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.d_value)
        return self.output_proj(attention)


class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            attn_pdrop: Optional[float] = None,
            residual_pdrop: Optional[float] = None) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        d_k = int(d_model // num_heads)
        self.attn = MultiHeadSelfAttention(num_heads, d_model, d_k, d_k, d_k, attn_pdrop, residual_pdrop)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.ffn = FFN(d_model, d_ff)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        y = x + self.attn(self.ln1(x))
        y = y + self.ffn(self.ln2(y))
        return y


class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            attn_pdrop: float,
            residual_pdrop: float) -> None:
        super().__init__()
        self.token_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.position_embeddings = nn.Embedding(num_embeddings=context_length, embedding_dim=d_model)
        self.dropout = nn.Dropout(residual_pdrop)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, attn_pdrop, residual_pdrop) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.context_length = context_length

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.context_length:
            raise ValueError(f"Input sequence length ({seq_len}) exceeds context_length ({self.context_length})")
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        output = self.dropout(self.token_embeddings(x) + self.position_embeddings(positions))
        for transformer_block in self.layers:   
            output = transformer_block(output)

        output = self.ln_final(output)
        output = self.lm_head(output)

        return output

class CrossEntropy(nn.Module):
    def forward(self, o: Float, targets: torch.LongTensor) -> torch.Tensor:
        # Numerical stability: subtract max
        o = o - torch.max(o, dim=-1, keepdim=True).values
        # Log-sum-exp for each example
        log_sum_exp = torch.log(torch.sum(torch.exp(o), dim=-1))
        # Get the logit for the correct class for each example
        correct_logits = o[torch.arange(o.shape[0])[:, None], torch.arange(o.shape[1])[None,:], targets]
        # correct_logits = o[torch.arange(o.shape[0]), targets]
        # Cross-entropy: log_sum_exp - correct_logit, averaged over batch
        return torch.mean(log_sum_exp - correct_logits)


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay: float=0.0, eps: float=1e-8) -> None:
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> None | float:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                # Initialize state on first step
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                t = state["t"] + 1
                m = state["m"]
                v = state["v"]

                grad = p.grad.data

                # Update biased first and second moment estimates
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad * grad

                # Bias correction
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                # Adam update
                p.data = p.data - lr * m_hat / (torch.sqrt(v_hat) + eps)

                # Weight decay (decoupled)
                p.data = p.data - lr * weight_decay * p.data

                state["t"] = t
                state["m"] = m
                state["v"] = v
        return loss


def cosine_schedule(t: int, alpha_min: float, alpha_max: float, Tw: float, Tc: float) -> float:
    import numpy as np
    if t < Tw:
        return t * alpha_max / Tw
    
    if t >= Tw and t < Tc:
        return alpha_min + .5 * (1 + np.cos(3.1415926 * (t - Tw) / (Tc - Tw))) * (alpha_max - alpha_min)

    return alpha_min


def clip_gradients(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float=1e-8) -> None:
    total_norm_sq = .0
    for parameter in parameters:
        grad = parameter.grad
        if grad is not None:
            parameter_norm_sq = torch.sum(grad ** 2)
            total_norm_sq += parameter_norm_sq
    total_norm = total_norm_sq ** .5
    if total_norm > max_l2_norm:
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad = parameter.grad * max_l2_norm / (total_norm + eps)


def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    max_starting_idx = len(x) - context_length
    ix = np.random.randint(0, max_starting_idx, size=(batch_size,))

    # Create lists to hold the slices
    x_list = [x[i : i + context_length] for i in ix]
    y_list = [x[i + 1 : i + context_length + 1] for i in ix]

    # Convert to tensors and move to device
    # Using np.stack is efficient for creating a (B, m) shape
    X = torch.from_numpy(np.stack(x_list)).to(device).long()
    Y = torch.from_numpy(np.stack(y_list)).to(device).long()

    return X, Y


def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]) -> None:
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(state, out)


def load_checkpoint(
        src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer) -> int:
    state_dict = torch.load(src)
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    return state_dict['iteration']

# TODO
def decode(tokenizer, model: torch.nn.Module, prompt: str, max_sequence_length: int) -> str:
    prompt_tokens = tokenizer.encode(prompt) 
    output = model(prompt)
