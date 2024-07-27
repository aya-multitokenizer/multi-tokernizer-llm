"""Multi-Tokenizer Model."""

from typing import Callable

import torch
import torch.nn as nn
from torch.nn import functional as F  # noqa: N812


class Head(nn.Module):
    """One Head of self-attention."""

    def __init__(self, c: int, h: int, t: int, dropout: float) -> None:
        """Initialize the head of self-attention."""
        super().__init__()
        self.C = c
        self.H = h
        self.T = t
        self.query = nn.Linear(self.C, self.H, bias=False)
        self.key = nn.Linear(self.C, self.H, bias=False)
        self.value = nn.Linear(self.C, self.H, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(self.T, self.T)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the head."""
        query_vectors = self.query(x)
        key_vectors = self.key(x)

        # Attention masking(so we can't look into the past):
        tril = self.tril
        wei = torch.zeros(self.T, self.T)
        wei = wei.masked_fill(
            tril == 0, float("-inf")
        )  # set the upper triangular to -inf

        # multiply the two to get the attention weights
        attention_pattern = query_vectors @ key_vectors.transpose(-2, -1)  # (t, t)
        attention_pattern = attention_pattern / (
            self.H**0.5
        )  # scale the attention pattern for numerical stability
        attention_weights = F.softmax(
            attention_pattern + wei, dim=-1
        )  # T, T (the row dimension is the query)
        attention_weights = self.dropout(attention_weights)

        # the direction we should go in the embedding space for each token (ie more blue) (t, h)
        value_vectors = self.value(x)

        # apply the attention weights to the value vectors
        context = attention_weights @ value_vectors  # (t, h)

        # project back into original space from value space
        return context


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention."""

    # H is head embedding space size, n_heads is number of heads
    def __init__(self, c: int, h: int, t: int, n_heads: int, dropout: float) -> None:
        """Initialize the multi-head attention."""
        super().__init__()
        self.heads = nn.ModuleList([Head(c, h, t, dropout) for _ in range(n_heads)])
        self.combine_heads = nn.Linear(h * n_heads, c)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the multi-head attention."""
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.combine_heads(x)  # (t, c)
        return self.dropout(x)


class FeedForward(nn.Module):
    """Feed-forward neural network."""

    def __init__(self, c: int, feedforward_factor: int, dropout: float) -> None:
        """Initialize the feed-forward neural network."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(c, c * feedforward_factor),
            nn.ReLU(),
            nn.Linear(c * feedforward_factor, c),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the feed forward neural network."""
        return self.net(x)


class LayerNorm(nn.Module):
    """Layer normalization."""

    def __init__(self, c: int, use_affine: bool = True) -> None:
        """Initialize the layer normalization."""
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(c)) if use_affine else None
        self.beta = nn.Parameter(torch.zeros(c)) if use_affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the layer normalization."""
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        if self.gamma is not None and self.beta is not None:
            return self.gamma * (x - mean) / (std + 1e-6) + self.beta
        else:
            return (x - mean) / (std + 1e-6)


class Block(nn.Module):
    """Transformer block."""

    def __init__(
        self,
        c: int,
        h: int,
        t: int,
        n_heads: int,
        dropout: float,
        feedforward_factor: int,
    ) -> None:
        """Initialize the transformer block."""
        super().__init__()
        self.attention = MultiHeadAttention(c, h, t, n_heads, dropout)
        self.ff = FeedForward(c, feedforward_factor, dropout)
        self.norm1 = LayerNorm(c, use_affine=True)
        self.norm2 = LayerNorm(c, use_affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the transformer block."""
        x = x + self.attention(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class MultiTokenizerLLM(nn.Module):
    """Multi-tokenizer language model."""

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        c: int,
        h: int,
        t: int,
        dropout: float,
        feedforward_factor: int,
        vocab_size: int,
    ) -> None:
        """Initialize the multi-tokenizer language model."""
        super().__init__()
        self.vocab_size = vocab_size
        self.T = t
        self.token_embedding_table = nn.Embedding(vocab_size, c)
        self.position_embedding_table = nn.Embedding(self.T, c)
        self.lm_head = nn.Linear(c, vocab_size)
        self.layers = nn.ModuleList(
            [
                Block(c, h, t, n_heads, dropout, feedforward_factor)
                for _ in range(n_layers)
            ]
        )
        self.block = nn.ModuleList(
            [Block(c, h, t, n_heads, dropout, feedforward_factor)]
        )

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of the multi-tokenizer language model."""
        token_emb = self.token_embedding_table(
            idx
        )  # batch_dim, sequence_dim, embedding_dim
        pos_emb = self.position_embedding_table(torch.arange(self.T))
        x = token_emb + pos_emb  # token identities and positions contained

        for layer in self.layers:
            x = layer(x)

        logits = self.lm_head(x)  # batch_dim, sequence_dim, vocab_size

        batch_dim, sequence_dim, embedding_dim = logits.size()

        if targets is None:
            return logits, None
        else:
            logits_loss_view = logits.view(-1, self.vocab_size)
            targets_loss_view = targets.view(-1)
            loss = F.cross_entropy(logits_loss_view, targets_loss_view)
            return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Generate new tokens."""
        for _ in range(max_new_tokens):
            logits, loss = self(idx[:, -self.T :])
            # get the predictions of the last token
            last_token_logits = logits[
                :, -1, :
            ]  # all batches, last token, all probabilities
            # softmax to get probabilities
            probabilities = F.softmax(last_token_logits, dim=-1)
            # sample from the probabilities
            next_token = torch.multinomial(probabilities, num_samples=1)
            # add the new token to the idx tensor
            idx = torch.cat((idx, next_token), dim=1)
        return idx

    def prompt_model(
        self,
        prompt: str,
        max_new_tokens: int,
        model: "MultiTokenizerLLM",
        encoder: Callable,
        decoder: Callable,
        temperature: float = 0.5,
    ) -> str:
        """Generate new tokens."""
        autoregressive_seq = encoder(prompt)
        for _ in range(max_new_tokens):
            prediction_index = len(autoregressive_seq) - 1

            model_input = torch.tensor(autoregressive_seq)

            while model_input.shape[0] < self.T:
                pad_token = torch.tensor(encoder("\n"))
                model_input = torch.cat((model_input, pad_token), dim=0)

            model_input
            model_input = model_input.unsqueeze(0)

            logits, loss = model(model_input)
            prediction_token = logits[:, prediction_index, :] / temperature
            probabilities = F.softmax(prediction_token, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            next_token = next_token.item()

            autoregressive_seq.append(next_token)
        # get the autoregressive sequence
        return decoder(autoregressive_seq)
