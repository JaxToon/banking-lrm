"""SASRec (self-attentive sequential recommendation) and ablations, synthetic-data research only.

Full variant (Kang & McAuley, 2018, adapted to this project):
1. Item embeddings (index 0 = padding, fixed at zero). A history step that holds
   several tied items is embedded as the masked MEAN of their item embeddings, so
   tied items are never ordered.
2. Learned positional embeddings over array positions. Sequences are left-padded,
   so the most recent step always sits at the last position.
3. `blocks` pre-LayerNorm Transformer blocks: explicit multi-head self-attention with
   a causal mask (position i attends to positions <= i) and a key-padding mask, then a
   position-wise feed-forward network (ReLU), each with dropout and a residual path.
   Padded positions are re-zeroed after every block. Attention is written with plain
   matmul/softmax (no fused SDPA kernels) to stay deterministic on CUDA.
4. Final LayerNorm; the representation of the last valid position (always the last
   array position under left padding) is scored against the item embeddings (tied
   weights, full softmax over the task's candidate items).

Ablations (each changes exactly one component; everything else is shared code):
- no_position: no positional embedding table. Causal masking is kept, so the model is
  NOT fully order-invariant (stacked causal layers still see prefix structure).
- shuffled: the Full architecture; only TRAINING batches have the order of their
  historical steps permuted (seeded, see shuffle_history). Validation and test scoring
  use the chronological order.
- meanpool: no Transformer blocks and no positional embeddings: the representation is
  the masked mean of the step embeddings over valid steps (padding excluded), followed
  by the same final LayerNorm and the same tied-embedding scoring.
- session_boundary: Full plus a 2-entry boundary embedding added to each step
  (new_session flag 0/1 from the historical prefix only, see bri.seq.data).

A case with an empty history gets all-zero scores, so the shared evaluator's
tie-break (train popularity, then item id) decides its ranking.

Loss (multi-positive full softmax): for positive set P,
    loss = -log(sum_{p in P} softmax(s)_p) = logsumexp(s) - logsumexp(s_P),
which equals standard cross-entropy when |P| = 1.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

MODEL_VERSION = "0.2.0"  # 0.2.0: ablation variants
VARIANTS = ("full", "no_position", "shuffled", "meanpool", "session_boundary")
# Hyperparameters a variant does not use (recorded, not searched separately).
UNUSED_PARAMS = {"full": (), "no_position": (), "shuffled": (), "session_boundary": (),
                 "meanpool": ("heads", "blocks")}


@dataclass(frozen=True)
class SASRecConfig:
    n_items: int
    max_len: int
    dim: int = 32
    heads: int = 2
    blocks: int = 2
    dropout: float = 0.2
    variant: str = "full"

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if self.n_items < 1 or self.max_len < 1 or self.blocks < 1 or self.heads < 1:
            raise ValueError("n_items, max_len, blocks and heads must be >= 1")
        if self.dim % self.heads:
            raise ValueError(f"dim ({self.dim}) must be divisible by heads ({self.heads})")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def uses_position(self) -> bool:
        return self.variant in ("full", "shuffled", "session_boundary")

    @property
    def uses_attention(self) -> bool:
        return self.variant != "meanpool"

    @property
    def uses_session_boundary(self) -> bool:
        return self.variant == "session_boundary"

    @property
    def shuffles_training_history(self) -> bool:
        return self.variant == "shuffled"

    def to_dict(self) -> dict:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.d_head = heads, dim // heads
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim) for _ in range(4))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        b, length, d = x.shape

        def split(t):
            return t.view(b, length, self.heads, self.d_head).transpose(1, 2)  # (b, h, L, dh)

        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)  # (b, h, L, L)
        causal = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
        allowed = causal & valid[:, None, None, :]  # keys must be real steps at or before the query
        allowed = allowed | torch.eye(length, dtype=torch.bool, device=x.device)  # no empty rows
        attn = torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1)
        out = (self.drop(attn) @ v).transpose(1, 2).reshape(b, length, d)
        return self.o(out)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, heads, dropout)
        self.ffn = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x), valid))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x * valid[..., None]


class SASRec(nn.Module):
    """All variants share one construction path.

    Every component is created and initialized in the same order for every variant
    (same random draws), then the components a variant does not use are dropped. So
    for one seed, the parameters a variant shares with Full start bitwise identical,
    and the variant has no dead parameters.
    """

    def __init__(self, cfg: SASRecConfig):
        super().__init__()
        self.cfg = cfg
        self.item_emb = nn.Embedding(cfg.n_items + 1, cfg.dim, padding_idx=0)
        pos_emb = nn.Embedding(cfg.max_len, cfg.dim)
        boundary_emb = nn.Embedding(2, cfg.dim)
        self.emb_drop = nn.Dropout(cfg.dropout)
        blocks = nn.ModuleList(Block(cfg.dim, cfg.heads, cfg.dropout) for _ in range(cfg.blocks))
        self.ln_final = nn.LayerNorm(cfg.dim)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        nn.init.normal_(pos_emb.weight, std=0.02)
        nn.init.normal_(boundary_emb.weight, std=0.02)
        self.pos_emb = pos_emb if cfg.uses_position else None
        self.boundary_emb = boundary_emb if cfg.uses_session_boundary else None
        self.blocks = blocks if cfg.uses_attention else nn.ModuleList()

    def step_embeddings(self, items: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """items (b, L, T) -> (step embeddings (b, L, d), valid mask (b, L))."""
        present = (items > 0).float()  # (b, L, T)
        n = present.sum(-1)
        emb = (self.item_emb(items) * present[..., None]).sum(-2) / n.clamp(min=1.0)[..., None]
        return emb, n > 0

    def _inputs(self, items: torch.Tensor, new_session: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if items.shape[1] != self.cfg.max_len:
            raise ValueError(f"expected sequences of length {self.cfg.max_len}, got {items.shape[1]}")
        x, valid = self.step_embeddings(items)
        x = x * math.sqrt(self.cfg.dim)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(self.cfg.max_len, device=items.device))[None]
        if self.boundary_emb is not None:
            if new_session is None:
                raise ValueError("the session_boundary variant needs new_session flags")
            x = x + self.boundary_emb(new_session.long())
        return self.emb_drop(x * valid[..., None]), valid

    def encode(self, items: torch.Tensor, new_session: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Hidden states for every position (b, L, d) and the valid mask (b, L). Not used by meanpool."""
        x, valid = self._inputs(items, new_session)
        for block in self.blocks:
            x = block(x, valid)
        return self.ln_final(x), valid

    def representation(self, items: torch.Tensor, new_session: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """(b, d) case representation and (b,) non-empty mask."""
        if not self.cfg.uses_attention:  # meanpool: masked mean over valid steps
            x, valid = self._inputs(items, new_session)
            pooled = x.sum(1) / valid.sum(1).clamp(min=1)[:, None].float()
            return self.ln_final(pooled), valid.any(1)
        h, valid = self.encode(items, new_session)
        return h[:, -1, :], valid[:, -1]  # most recent step under left padding

    def forward(self, items: torch.Tensor, new_session: torch.Tensor | None = None) -> torch.Tensor:
        """Scores (b, n_items) over the task's candidate items."""
        rep, nonempty = self.representation(items, new_session)
        scores = rep @ self.item_emb.weight[1:].T
        return scores * nonempty[:, None].float()  # empty history -> all-zero scores


def shuffle_history(items: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Randomly permute the order of the valid (non-padding) steps of every sequence.

    Padding stays at the front (left padding), each step keeps all of its tied items,
    and the multiset of steps is unchanged. Random keys come from `generator` on the
    CPU, so the permutation is identical on every device for the same seed.
    """
    valid = (items > 0).any(-1)  # (b, L)
    keys = torch.rand(valid.shape, generator=generator).to(items.device)
    keys = torch.where(valid, keys, torch.full_like(keys, -1.0))  # padding sorts first
    order = keys.argsort(dim=1)
    return torch.gather(items, 1, order[..., None].expand_as(items))


def multi_positive_loss(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean over cases of -log(sum of softmax probabilities of the positive items)."""
    pos = targets > 0
    if not pos.any(dim=1).all():
        raise ValueError("every case needs at least one positive item")
    log_pos = torch.logsumexp(scores.masked_fill(~pos, float("-inf")), dim=1)
    return (torch.logsumexp(scores, dim=1) - log_pos).mean()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
