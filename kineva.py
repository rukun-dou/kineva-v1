"""Kineva: a pretrained encoder for surgical instrument kinematics."""

import torch
from torch import nn

__all__ = ["Kineva"]

# ---------------------------------------------------------------------------
# rotary position embeddings
# ---------------------------------------------------------------------------


class RotaryPositionalEmbeddings(nn.Module):
    """RoPE applied to the last two dimensions of the input.

    Args:
        dim: head dimension.
        max_seq_len: number of positions to precompute, extended if exceeded.
        base: base of the geometric progression of rotation angles.
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10_000) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError(f"dim must be even, got {dim}")
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        theta = 1.0 / (base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
        self.register_buffer("theta", theta, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, max_seq_len: int) -> None:
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

        self.register_buffer("cache", cache, persistent=False)
        self.max_seq_len = max_seq_len

    def forward(
        self, x: torch.Tensor, input_pos: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Rotate ``x`` of shape ``(batch, seq, heads, head_dim)`` at ``input_pos``."""
        if input_pos is None:
            if x.size(1) > self.max_seq_len:
                self._build_cache(max(2 * self.max_seq_len, x.size(1)))
            rope_cache = self.cache[: x.size(1)]
        else:
            if input_pos.numel() and int(input_pos.max()) >= self.max_seq_len:
                self._build_cache(max(2 * self.max_seq_len, int(input_pos.max()) + 1))
            rope_cache = self.cache[input_pos]

        x_shaped = x.float().reshape(*x.shape[:-1], -1, 2)
        rope_cache = rope_cache.view(-1, x_shaped.size(1), 1, x_shaped.size(3), 2)
        x_out = torch.stack(
            [
                x_shaped[..., 0] * rope_cache[..., 0]
                - x_shaped[..., 1] * rope_cache[..., 1],
                x_shaped[..., 1] * rope_cache[..., 0]
                + x_shaped[..., 0] * rope_cache[..., 1],
            ],
            dim=-1,
        )
        return x_out.flatten(3).type_as(x)


# ---------------------------------------------------------------------------
# patch embedding
# ---------------------------------------------------------------------------


class PatchEmbedding(nn.Module):
    """Progressive convolutional encoder folding ``patch_size`` timesteps into a token.

    Each patch is z-normalised per channel, then its mean and standard deviation
    are projected back in so that absolute scale survives the normalisation.
    """

    def __init__(self, embed_dim: int, patch_size: int = 16, eps: float = 1e-6):
        super().__init__()
        assert (
            patch_size > 0 and (patch_size & (patch_size - 1)) == 0
        ), "patch_size must be a power of 2"

        self.patch_size = patch_size
        self.eps = eps

        # strides widen towards deeper stages, capped at 3 halvings
        strides = []
        rem = patch_size
        if patch_size == 1:
            strides = [1, 1]
        else:
            while rem > 1:
                if rem % 2 == 0 and len(strides) < 3:
                    strides.append(2)
                    rem //= 2
                else:
                    strides.append(rem)
                    break
        num_layers = len(strides)

        channels = [1] + [
            int(embed_dim * (i + 1) / num_layers) for i in range(num_layers)
        ]
        channels[-1] = embed_dim

        stages = []
        c_in = 1
        for i, s in enumerate(strides):
            c_out = channels[i + 1]
            k = 1 if s == 1 else 2 * s
            p = 0 if s == 1 else s - 1

            stages.append(nn.Conv1d(c_in, c_out, kernel_size=k, stride=s, padding=p))
            if i < num_layers - 1 and patch_size > 1:
                stages.append(nn.GroupNorm(1, c_out))
            stages.append(nn.GELU())
            c_in = c_out

        self.encoder = nn.Sequential(*stages)
        self.scale_proj = nn.Linear(2, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        orig_batch_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # pad up to a whole number of patches
        remainder = x.size(1) % self.patch_size
        if remainder:
            x = nn.functional.pad(x, (0, 0, 0, self.patch_size - remainder))
        T_padded = x.size(1)
        n_patches = T_padded // self.patch_size

        x_patches = x[..., 0].reshape(x.size(0), n_patches, self.patch_size)

        if orig_batch_size is not None:
            # normalise per channel independently within each patch
            B = orig_batch_size
            C = x.size(0) // B
            x_4d = x_patches.reshape(B, C, n_patches, self.patch_size)
            mu = x_4d.mean(dim=-1, keepdim=True)
            sigma = x_4d.std(dim=-1, keepdim=True).clamp(min=self.eps)
            x_norm = ((x_4d - mu) / sigma).reshape(B * C, n_patches, self.patch_size)
            mu = mu.reshape(B * C, n_patches, 1)
            sigma = sigma.reshape(B * C, n_patches, 1)
        else:
            mu = x_patches.mean(dim=-1, keepdim=True)
            sigma = x_patches.std(dim=-1, keepdim=True).clamp(min=self.eps)
            x_norm = (x_patches - mu) / sigma

        x_seq = x_norm.reshape(x.size(0), T_padded, 1)
        patch_emb = self.encoder(x_seq.transpose(1, 2)).transpose(1, 2)
        patch_emb = patch_emb + self.scale_proj(torch.cat([mu, sigma], dim=-1))

        patch_mask = None
        if key_padding_mask is not None:
            if key_padding_mask.size(1) < T_padded:
                extra = key_padding_mask.new_ones(
                    key_padding_mask.size(0), T_padded - key_padding_mask.size(1)
                )
                key_padding_mask = torch.cat([key_padding_mask, extra], dim=1)
            patch_mask = (
                key_padding_mask[:, :T_padded]
                .reshape(key_padding_mask.size(0), n_patches, self.patch_size)
                .all(-1)
            )

        return patch_emb, patch_mask


# ---------------------------------------------------------------------------
# cross-channel attention
# ---------------------------------------------------------------------------


class CrossChannelTransformerLayer(nn.Module):
    """Pre-norm transformer layer over the channel axis."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, N, D = x.shape
        normed = self.norm1(x)

        qkv = (
            self.qkv(normed)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = torch.zeros(B, 1, 1, N, dtype=q.dtype, device=q.device)
            attn_mask.masked_fill_(key_padding_mask[:, None, None, :], float("-inf"))

        attn_out = (
            nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
            .transpose(1, 2)
            .reshape(B, N, D)
        )

        x = x + self.drop(self.out_proj(attn_out))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class AttentionPooling(nn.Module):
    """Pool the channel axis of one patch into a single token with learned queries.

    RoPE is applied to the keys so that pooling sees channel order, unlike the
    permutation-equivariant layers that precede it.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        n_token: int = 4,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.n_token = n_token

        self.latent_query = nn.Parameter(torch.zeros(1, n_token, embed_dim))
        nn.init.trunc_normal_(self.latent_query, std=0.02)

        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
        )
        self.drop = nn.Dropout(dropout)
        self.bottleneck_proj = nn.Linear(n_token * embed_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryPositionalEmbeddings,
        positions: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, D = x.shape

        # the latent query is shared across the flattened batch dimension
        q_lat = self.latent_query.expand(B, -1, -1)
        normed_q = self.norm_q(q_lat)
        normed_kv = self.norm_kv(x)

        q = (
            self.q_proj(normed_q)
            .reshape(B, self.n_token, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        kv = (
            self.kv_proj(normed_kv)
            .reshape(B, N, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)

        # rotate keys only; the queries carry no channel position
        k = k.permute(0, 2, 1, 3)
        k = rope(k, input_pos=positions)
        k = k.permute(0, 2, 1, 3)

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = torch.zeros(B, 1, 1, N, dtype=q.dtype, device=q.device)
            attn_mask.masked_fill_(key_padding_mask[:, None, None, :], float("-inf"))

        attn_out = (
            nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
            .transpose(1, 2)
            .reshape(B, self.n_token, D)
        )

        q_lat = q_lat + self.drop(self.out_proj(attn_out))
        q_lat = q_lat + self.drop(self.ff(self.norm2(q_lat)))

        return self.bottleneck_proj(q_lat.reshape(B, self.n_token * D))


class CrossChannelAttention(nn.Module):
    """Mix information across channels within a patch, then pool to one token."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        num_layers: int = 3,
        n_token: int = 4,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CrossChannelTransformerLayer(embed_dim, num_heads, mlp_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.pooling = AttentionPooling(embed_dim, num_heads, mlp_dim, dropout, n_token)
        self.rope = RotaryPositionalEmbeddings(
            dim=embed_dim // num_heads, max_seq_len=512
        )

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, N, _ = x.shape
        positions = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        return self.pooling(x, self.rope, positions, key_padding_mask)


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------


class Tokenizer(nn.Module):
    """Fold ``(batch, timesteps, channels)`` kinematics into one token per patch.

    Channels share the patch encoder, then exchange information within each patch
    and are pooled. Patch timestamps are averaged and quantised into position indices.
    """

    def __init__(
        self,
        embed_dim: int,
        patch_size: int,
        num_heads: int,
        base_time_tick_hz: float,
        mlp_dim: int = 1536,
        dropout: float = 0.15,
        num_cc_layers: int = 3,
        n_token: int = 4,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.base_time_tick_hz = base_time_tick_hz
        self.patch_embed = PatchEmbedding(embed_dim, patch_size)
        self.cross_channel = CrossChannelAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            num_layers=num_cc_layers,
            n_token=n_token,
        )

    def forward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        channel_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        B, T, C = x.shape

        if channel_padding_mask is None:
            channel_padding_mask = torch.zeros(B, C, dtype=torch.bool, device=x.device)
        elif channel_padding_mask.size(1) < C:
            extra = channel_padding_mask.new_ones(B, C - channel_padding_mask.size(1))
            channel_padding_mask = torch.cat([channel_padding_mask, extra], dim=1)
        else:
            channel_padding_mask = channel_padding_mask[:, :C]

        # fold channels into the batch dimension to share the patch encoder
        x_ch = x.permute(0, 2, 1).reshape(B * C, T, 1)
        km_ch = None
        if key_padding_mask is not None:
            km_ch = key_padding_mask.unsqueeze(1).expand(B, C, T).reshape(B * C, T)

        x_ch, patch_mask_ch = self.patch_embed(x_ch, km_ch, orig_batch_size=B)
        n_patches = x_ch.size(1)
        embed_dim = x_ch.size(2)

        tokens = x_ch.reshape(B, C, n_patches, embed_dim).permute(0, 2, 1, 3)
        valid_channels = ~channel_padding_mask
        if patch_mask_ch is not None:
            patch_channel_padding = patch_mask_ch.reshape(B, C, n_patches).permute(
                0, 2, 1
            )
            valid = (~patch_channel_padding) & valid_channels[:, None, :]
        else:
            valid = valid_channels[:, None, :].expand(B, n_patches, C)

        tokens = tokens.reshape(B * n_patches, C, embed_dim)
        cc_mask = (~valid).reshape(B * n_patches, C)
        all_masked = cc_mask.all(dim=1)

        # patches whose channels are all invalid still need a finite attention row
        fully_masked_patches = all_masked.reshape(B, n_patches)
        if all_masked.any():
            cc_mask[all_masked, 0] = False

        tokens = self.cross_channel(tokens, cc_mask)
        tokens = tokens.reshape(B, n_patches, embed_dim)

        # pad timestamps to the padded length before averaging per patch
        T_padded = n_patches * self.patch_size
        if timestamps.size(1) < T_padded:
            pad_times = timestamps[:, -1:].expand(B, T_padded - timestamps.size(1))
            ts_padded = torch.cat([timestamps, pad_times], dim=1)
        else:
            ts_padded = timestamps[:, :T_padded]

        mean_ts = ts_padded.reshape(B, n_patches, self.patch_size).mean(dim=-1)

        # elapsed time relative to the first patch, quantised to integer ticks
        tick_seconds = 1.0 / self.base_time_tick_hz
        positions = torch.round(
            (mean_ts - mean_ts[:, :1]).clamp(min=0.0) / tick_seconds
        ).long()

        token_mask = (
            patch_mask_ch.reshape(B, C, n_patches).all(dim=1)
            if patch_mask_ch is not None
            else None
        )
        if token_mask is not None:
            token_mask = token_mask | fully_masked_patches
            if not token_mask.any():
                token_mask = None
        elif fully_masked_patches.any():
            token_mask = fully_masked_patches

        return tokens, token_mask, positions


# ---------------------------------------------------------------------------
# temporal attention
# ---------------------------------------------------------------------------


class RoPEMultiheadAttention(nn.Module):
    """Multi-head attention with rotary embeddings on Q and K."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryPositionalEmbeddings,
        positions: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, D = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)

        # the CLS token stays unrotated; RoPE applies to temporal tokens only
        q_cls, q_tok = q[:, :1], q[:, 1:]
        k_cls, k_tok = k[:, :1], k[:, 1:]
        q_tok = rope(q_tok, input_pos=positions[:, 1:])
        k_tok = rope(k_tok, input_pos=positions[:, 1:])
        q = torch.cat([q_cls, q_tok], dim=1).permute(0, 2, 1, 3)
        k = torch.cat([k_cls, k_tok], dim=1).permute(0, 2, 1, 3)

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = torch.zeros(B, 1, 1, N, dtype=q.dtype, device=q.device)
            attn_mask.masked_fill_(key_padding_mask[:, None, None, :], float("-inf"))

        out = (
            nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
            .transpose(1, 2)
            .reshape(B, N, D)
        )
        return self.out_proj(out)


class RoPETransformerEncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer using RoPE attention."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = RoPEMultiheadAttention(embed_dim, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryPositionalEmbeddings,
        positions: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), rope, positions, key_padding_mask))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class Kineva(nn.Module):
    """Kineva backbone.

    A channel-shared convolutional encoder folds 16 timesteps into one token per
    channel. Three cross-channel transformer layers let channels interact within a
    patch, and four learned queries pool them into a single token. Eight temporal
    transformer layers then run over the patch sequence with rotary position
    embeddings. A prepended CLS token aggregates the procedure.

    Channel identity, order and count are never encoded, and positions come from
    the timestamps, so the same weights apply to any sensor configuration.

    Args:
        embed_dim: token width.
        patch_size: timesteps per patch, must be a power of two.
        num_layers: temporal transformer layers.
        num_heads: attention heads.
        mlp_dim: hidden width of the feed-forward blocks.
        dropout: dropout rate, inactive at inference.
        base_time_tick_hz: resolution of the quantised timestamps.
        num_cc_layers: cross-channel transformer layers.
        n_token: learned pooling queries per patch.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        patch_size: int = 16,
        num_layers: int = 8,
        num_heads: int = 8,
        mlp_dim: int = 1536,
        dropout: float = 0.15,
        base_time_tick_hz: float = 5.0,
        num_cc_layers: int = 3,
        n_token: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.tokenizer = Tokenizer(
            embed_dim,
            patch_size,
            num_heads,
            base_time_tick_hz,
            mlp_dim=mlp_dim,
            dropout=dropout,
            num_cc_layers=num_cc_layers,
            n_token=n_token,
        )
        self.rope = RotaryPositionalEmbeddings(
            dim=embed_dim // num_heads, max_seq_len=32768
        )
        self.layers = nn.ModuleList(
            [
                RoPETransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.cls_token_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))

    @classmethod
    def from_pretrained(
        cls,
        path: str = "kineva_weights.pth",
        device: str | torch.device = "cpu",
        **kwargs,
    ) -> "Kineva":
        """Load the released weights.

        Args:
            path: checkpoint file.
            device: device to place the model on.
            **kwargs: overrides for the configuration above.

        Returns:
            A frozen model in evaluation mode.
        """
        model = cls(**kwargs)
        state = torch.load(path, map_location="cpu", weights_only=True)

        # tolerate checkpoints that wrap the weights under a key
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
        return model.to(device).eval().requires_grad_(False)

    def encode(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        channel_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Global representation of each recording, shape ``(batch, embed_dim)``.

        Args:
            x: kinematics, ``(batch, timesteps, channels)``.
            timestamps: seconds, ``(batch, timesteps)``. Need not be regular.
            key_padding_mask: ``(batch, timesteps)``, True marks padded timesteps.
            channel_padding_mask: ``(batch, channels)``, True marks absent channels.
                Use this when a recording carries fewer channels than another in the
                same batch.
        """
        return self._run(x, timestamps, key_padding_mask, channel_padding_mask)[:, 0]

    def encode_patches(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        channel_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-patch tokens, shape ``(batch, n_patches, embed_dim)``.

        One token per 16 timesteps, in temporal order.
        """
        return self._run(x, timestamps, key_padding_mask, channel_padding_mask)[:, 1:]

    def _run(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        channel_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return the normalised sequence of CLS followed by the patch tokens."""
        x, timestamps, key_padding_mask, channel_padding_mask = self._prepare(
            x, timestamps, key_padding_mask, channel_padding_mask
        )
        with torch.no_grad():
            self._check_timestamps(x, timestamps)
            tokens, token_mask, patch_positions = self.tokenizer(
                x, timestamps, key_padding_mask, channel_padding_mask
            )
            B = tokens.size(0)

            # the CLS token is prepended and left unrotated
            cls_position = torch.zeros(B, 1, dtype=torch.long, device=x.device)
            positions = torch.cat([cls_position, patch_positions], dim=1)

            cls = self.cls_token_embed.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            if token_mask is not None:
                token_mask = torch.cat([token_mask.new_zeros(B, 1), token_mask], dim=1)

            for layer in self.layers:
                tokens = layer(tokens, self.rope, positions, token_mask)

        return self.norm(tokens)

    def _prepare(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        channel_padding_mask: torch.Tensor | None,
    ) -> tuple:
        device = self.cls_token_embed.device
        x = x.to(device).float()
        timestamps = timestamps.to(device).float()
        if x.dim() != 3:
            raise ValueError(
                f"x must be (batch, timesteps, channels), got {tuple(x.shape)}"
            )
        if timestamps.shape != x.shape[:2]:
            raise ValueError(
                f"timestamps must be (batch, timesteps) = {tuple(x.shape[:2])}, "
                f"got {tuple(timestamps.shape)}"
            )
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(device).bool()
        if channel_padding_mask is not None:
            channel_padding_mask = channel_padding_mask.to(device).bool()
        return x, timestamps, key_padding_mask, channel_padding_mask

    def _check_timestamps(self, x: torch.Tensor, timestamps: torch.Tensor) -> None:
        # handle the case where the recording is shorter than one patch
        patch_size = self.tokenizer.patch_size
        n_patches = (x.size(1) + patch_size - 1) // patch_size
        if n_patches < 2:
            raise ValueError(
                f"need at least 2 patches ({2 * patch_size} timesteps) to encode "
                f"a recording, got {x.size(1)}"
            )
        if not torch.isfinite(timestamps).all():
            raise ValueError("timestamps contain non-finite values")
        if (timestamps[:, 1:] < timestamps[:, :-1]).any():
            raise ValueError("timestamps must be non-decreasing")


if __name__ == "__main__":
    torch.manual_seed(0)
    model = Kineva.from_pretrained()
    print(
        model.__class__.__name__, sum(p.numel() for p in model.parameters()) / 1e6, "M"
    )

    x = torch.randn(2, 800, 67)
    t = torch.arange(800).float().repeat(2, 1) * 0.2
    print("cls      ", tuple(model.encode(x, t).shape))
    print("patches  ", tuple(model.encode_patches(x, t).shape))
