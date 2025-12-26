# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DETR-style dynamics head for per-object state prediction.

This module implements a transformer decoder that uses learnable object queries
to detect and predict dynamics (position, velocity, acceleration) for each
object in the scene across all frames.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


@dataclass
class DynamicsHeadConfig:
    """Configuration for DynamicsHead.

    Attributes:
        num_object_queries: Number of learnable object queries (max agents)
        hidden_dim: Hidden dimension for transformer decoder
        num_decoder_layers: Number of transformer decoder layers
        num_heads: Number of attention heads
        dim_feedforward: Feedforward dimension in transformer
        dropout: Dropout rate
        num_classes: Number of object classes (+ 1 for no-object)
        state_dim: Dimension of dynamics state (position + velocity + accel)
        num_frames: Number of frames for temporal processing
    """
    num_object_queries: int = 32
    hidden_dim: int = 256
    num_decoder_layers: int = 6
    num_heads: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.1
    num_classes: int = 4  # vehicle, pedestrian, cyclist, other
    state_dim: int = 9  # x, y, z, vx, vy, vz, ax, ay, az


class PositionalEncoding3D(nn.Module):
    """3D positional encoding for spatio-temporal features."""

    def __init__(self, hidden_dim: int, max_temporal: int = 128, max_spatial: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Learnable embeddings for T, H, W
        self.temporal_embed = nn.Embedding(max_temporal, hidden_dim // 3)
        self.height_embed = nn.Embedding(max_spatial, hidden_dim // 3)
        self.width_embed = nn.Embedding(max_spatial, hidden_dim // 3 + hidden_dim % 3)

    def forward(self, T: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Generate 3D positional encoding.

        Args:
            T: Number of temporal positions
            H: Height positions
            W: Width positions
            device: Target device

        Returns:
            Positional encoding tensor [T*H*W, hidden_dim]
        """
        t_pos = torch.arange(T, device=device)
        h_pos = torch.arange(H, device=device)
        w_pos = torch.arange(W, device=device)

        t_emb = self.temporal_embed(t_pos)  # [T, D//3]
        h_emb = self.height_embed(h_pos)    # [H, D//3]
        w_emb = self.width_embed(w_pos)     # [W, D//3+]

        # Create 3D grid
        t_grid = t_emb[:, None, None, :].expand(T, H, W, -1)
        h_grid = h_emb[None, :, None, :].expand(T, H, W, -1)
        w_grid = w_emb[None, None, :, :].expand(T, H, W, -1)

        # Concatenate along feature dimension
        pos_encoding = torch.cat([t_grid, h_grid, w_grid], dim=-1)  # [T, H, W, D]
        pos_encoding = rearrange(pos_encoding, 't h w d -> (t h w) d')

        return pos_encoding


class ObjectQueryEmbedding(nn.Module):
    """Learnable object query embeddings with temporal awareness."""

    def __init__(self, num_queries: int, hidden_dim: int, num_frames: int = 32):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.num_frames = num_frames

        # Learnable object queries [num_queries, hidden_dim]
        self.object_queries = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)

        # Temporal position encoding for each frame
        self.temporal_pos = nn.Parameter(torch.randn(num_frames, hidden_dim) * 0.02)

    def forward(self, batch_size: int, num_frames: int) -> torch.Tensor:
        """Generate object query embeddings for batch.

        Args:
            batch_size: Batch size
            num_frames: Number of frames

        Returns:
            Query embeddings [B, num_queries * num_frames, hidden_dim]
        """
        # Expand queries to batch
        queries = self.object_queries[None, :, None, :]  # [1, Q, 1, D]
        queries = queries.expand(batch_size, -1, num_frames, -1)  # [B, Q, T, D]

        # Add temporal position
        t_pos = self.temporal_pos[:num_frames]  # [T, D]
        queries = queries + t_pos[None, None, :, :]  # [B, Q, T, D]

        # Flatten queries
        queries = rearrange(queries, 'b q t d -> b (q t) d')

        return queries


class TransformerDecoderLayer(nn.Module):
    """Custom transformer decoder layer with self-attention and cross-attention."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

        # Feedforward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        query_pos: Optional[torch.Tensor] = None,
        memory_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            query: Query embeddings [B, Q, D]
            memory: Encoder memory (backbone features) [B, L, D]
            query_pos: Query positional encoding [B, Q, D]
            memory_pos: Memory positional encoding [B, L, D]

        Returns:
            Updated query embeddings [B, Q, D]
        """
        # Self-attention with positional encoding
        q = k = query + query_pos if query_pos is not None else query
        attn_output, _ = self.self_attn(q, k, query)
        query = query + self.dropout1(attn_output)
        query = self.norm1(query)

        # Cross-attention with memory
        q = query + query_pos if query_pos is not None else query
        k = memory + memory_pos if memory_pos is not None else memory
        attn_output, _ = self.cross_attn(q, k, memory)
        query = query + self.dropout2(attn_output)
        query = self.norm2(query)

        # Feedforward
        query = query + self.ffn(query)
        query = self.norm3(query)

        return query


class DynamicsHead(nn.Module):
    """DETR-style head for per-object dynamics prediction.

    This module takes intermediate features from a DiT backbone and predicts
    per-object dynamics (position, velocity, acceleration) and class for each
    frame in the sequence.
    """

    def __init__(self, config: DynamicsHeadConfig, backbone_dim: int):
        """Initialize DynamicsHead.

        Args:
            config: Head configuration
            backbone_dim: Dimension of backbone features (model_channels from DiT)
        """
        super().__init__()
        self.config = config
        self.backbone_dim = backbone_dim
        self.hidden_dim = config.hidden_dim

        # Project backbone features to hidden dimension
        self.input_proj = nn.Linear(backbone_dim, config.hidden_dim)

        # Object query embeddings
        self.query_embed = ObjectQueryEmbedding(
            config.num_object_queries,
            config.hidden_dim,
            num_frames=32,  # Max frames
        )

        # Positional encoding for backbone features
        self.pos_encoding = PositionalEncoding3D(config.hidden_dim)

        # Transformer decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                config.hidden_dim,
                config.num_heads,
                config.dim_feedforward,
                config.dropout,
            )
            for _ in range(config.num_decoder_layers)
        ])

        # Output heads
        # Class prediction: num_classes + 1 (for no-object)
        self.class_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes + 1),
        )

        # State prediction: position + velocity + acceleration
        self.state_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.state_dim),
        )

        # Confidence/objectness score
        self.confidence_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Initialize class head bias for no-object class
        # This encourages the model to predict no-object by default
        nn.init.constant_(self.class_head[-1].bias[-1], 2.0)

    def forward(
        self,
        backbone_features: List[torch.Tensor],
        num_frames: int,
        spatial_shape: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            backbone_features: List of intermediate features from DiT backbone
                Each tensor has shape [B, T*H*W, D] where D is backbone_dim
            num_frames: Number of temporal frames
            spatial_shape: Optional (H, W) spatial dimensions. If None, inferred from features.

        Returns:
            Dictionary with:
                - class_logits: [B, T, num_queries, num_classes+1]
                - state_pred: [B, T, num_queries, state_dim]
                - confidence: [B, T, num_queries, 1]
                - object_embeddings: [B, T, num_queries, hidden_dim]
        """
        # Concatenate features from multiple backbone layers
        # Shape: [B, T*H*W, D]
        if len(backbone_features) > 1:
            features = sum(backbone_features) / len(backbone_features)
        else:
            features = backbone_features[0]

        B, L, D = features.shape

        # Infer spatial dimensions
        if spatial_shape is None:
            # Assume square spatial dimensions
            spatial_size = int((L / num_frames) ** 0.5)
            H = W = spatial_size
        else:
            H, W = spatial_shape

        # Project to hidden dimension
        features = self.input_proj(features)  # [B, T*H*W, hidden_dim]

        # Generate positional encoding
        pos_encoding = self.pos_encoding(num_frames, H, W, features.device)
        pos_encoding = pos_encoding[None, :, :].expand(B, -1, -1)  # [B, T*H*W, hidden_dim]

        # Generate object queries
        queries = self.query_embed(B, num_frames)  # [B, Q*T, hidden_dim]
        query_pos = queries.clone()  # Use queries as positional encoding too

        # Decoder forward
        for layer in self.decoder_layers:
            queries = layer(queries, features, query_pos, pos_encoding)

        # Reshape queries to [B, num_queries, num_frames, hidden_dim]
        Q = self.config.num_object_queries
        queries = rearrange(queries, 'b (q t) d -> b q t d', q=Q, t=num_frames)

        # Predict outputs
        class_logits = self.class_head(queries)  # [B, Q, T, num_classes+1]
        state_pred = self.state_head(queries)    # [B, Q, T, state_dim]
        confidence = self.confidence_head(queries)  # [B, Q, T, 1]

        # Rearrange to [B, T, Q, D] format expected by loss functions
        class_logits = rearrange(class_logits, 'b q t c -> b t q c')
        state_pred = rearrange(state_pred, 'b q t s -> b t q s')
        confidence = rearrange(confidence, 'b q t 1 -> b t q 1')
        object_embeddings = rearrange(queries, 'b q t d -> b t q d')

        return {
            'class_logits': class_logits,
            'state_pred': state_pred,
            'confidence': confidence,
            'object_embeddings': object_embeddings,
        }

    def forward_single_scale(
        self,
        features: torch.Tensor,
        num_frames: int,
        spatial_shape: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with single-scale features.

        Convenience method when using features from a single backbone layer.

        Args:
            features: Backbone features [B, T*H*W, D]
            num_frames: Number of temporal frames
            spatial_shape: (H, W) spatial dimensions

        Returns:
            Same as forward()
        """
        return self.forward([features], num_frames, spatial_shape)
