# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamics embedder for conditioning world model on agent dynamics.

This module provides embeddings for dynamics conditioning, enabling:
1. Counterfactual generation by modifying agent trajectories
2. Physics-grounded predictions by conditioning on initial dynamics

The embedder follows the same pattern as action conditioning in
ActionConditionedMinimalV1LVGDiT, adding embeddings to both the
timestep embedding and AdaLN LoRA tensors.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange


class Mlp(nn.Module):
    """Multi-layer perceptron for embedding projection."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DynamicsEmbedder(nn.Module):
    """Embedder for per-object dynamics conditioning.

    Takes per-object dynamics (position, velocity, acceleration) and produces
    embeddings that can be added to the timestep embedding and AdaLN LoRA
    for conditioning the diffusion model.

    This enables:
    1. Conditioning generation on observed dynamics
    2. Counterfactual editing by modifying specific object trajectories

    Attributes:
        max_agents: Maximum number of agents to embed
        state_dim: Dimension of per-agent state (default 9: xyz + vel + accel)
        model_channels: Output dimension matching DiT model channels
    """

    def __init__(
        self,
        max_agents: int,
        state_dim: int,
        model_channels: int,
        num_frames: int = 32,
        hidden_multiplier: int = 4,
    ):
        """Initialize DynamicsEmbedder.

        Args:
            max_agents: Maximum number of agents
            state_dim: Dimension of per-agent state
            model_channels: DiT model channel dimension
            num_frames: Number of frames for temporal dynamics
            hidden_multiplier: Multiplier for hidden layer size
        """
        super().__init__()
        self.max_agents = max_agents
        self.state_dim = state_dim
        self.model_channels = model_channels
        self.num_frames = num_frames

        # Input dimension: flatten all agents' states across time
        # For counterfactual conditioning, we use a subset of frames
        input_dim = max_agents * state_dim

        # Embedder for timestep embedding (model_channels output)
        self.dynamics_embedder_D = Mlp(
            in_features=input_dim,
            hidden_features=model_channels * hidden_multiplier,
            out_features=model_channels,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )

        # Embedder for AdaLN LoRA (3 * model_channels output)
        self.dynamics_embedder_3D = Mlp(
            in_features=input_dim,
            hidden_features=model_channels * hidden_multiplier,
            out_features=model_channels * 3,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )

        # Optional: per-frame embedding for temporal dynamics
        self.use_temporal = False  # Can enable for frame-specific conditioning

    def forward(
        self,
        dynamics: torch.Tensor,
        dynamics_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute dynamics embeddings.

        Args:
            dynamics: Agent dynamics [B, T, max_agents, state_dim] or [B, max_agents, state_dim]
            dynamics_mask: Optional mask for valid agents [B, T, max_agents] or [B, max_agents]

        Returns:
            Tuple of:
                - t_emb: Embedding to add to timestep embedding [B, 1, model_channels]
                - adaln_emb: Embedding to add to AdaLN LoRA [B, 1, model_channels*3]
        """
        # Handle different input shapes
        if dynamics.dim() == 4:
            # [B, T, N, S] -> use first frame or aggregate
            B, T, N, S = dynamics.shape

            if dynamics_mask is not None:
                # Zero out invalid agents
                dynamics = dynamics * dynamics_mask.unsqueeze(-1)

            # Use first frame dynamics for conditioning (or could aggregate)
            dynamics = dynamics[:, 0]  # [B, N, S]

            if dynamics_mask is not None:
                dynamics_mask = dynamics_mask[:, 0]  # [B, N]
        else:
            # [B, N, S] - already single frame
            B, N, S = dynamics.shape
            if dynamics_mask is not None:
                dynamics = dynamics * dynamics_mask.unsqueeze(-1)

        # Flatten agent dimension
        dynamics_flat = rearrange(dynamics, 'b n s -> b (n s)')  # [B, N*S]

        # Compute embeddings
        t_emb = self.dynamics_embedder_D(dynamics_flat)  # [B, model_channels]
        adaln_emb = self.dynamics_embedder_3D(dynamics_flat)  # [B, model_channels*3]

        # Add temporal dimension for consistency with timestep embedding format
        t_emb = t_emb.unsqueeze(1)  # [B, 1, model_channels]
        adaln_emb = adaln_emb.unsqueeze(1)  # [B, 1, model_channels*3]

        return t_emb, adaln_emb


class TemporalDynamicsEmbedder(nn.Module):
    """Embedder for per-frame dynamics conditioning.

    Unlike DynamicsEmbedder which produces a single embedding,
    this version produces per-frame embeddings for fine-grained
    temporal conditioning.
    """

    def __init__(
        self,
        max_agents: int,
        state_dim: int,
        model_channels: int,
        num_frames: int = 32,
        hidden_multiplier: int = 4,
    ):
        """Initialize TemporalDynamicsEmbedder.

        Args:
            max_agents: Maximum number of agents
            state_dim: Dimension of per-agent state
            model_channels: DiT model channel dimension
            num_frames: Maximum number of frames
            hidden_multiplier: Multiplier for hidden layer size
        """
        super().__init__()
        self.max_agents = max_agents
        self.state_dim = state_dim
        self.model_channels = model_channels
        self.num_frames = num_frames

        input_dim = max_agents * state_dim

        # Per-frame embedders
        self.frame_embedder_D = Mlp(
            in_features=input_dim,
            hidden_features=model_channels * hidden_multiplier,
            out_features=model_channels,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )

        self.frame_embedder_3D = Mlp(
            in_features=input_dim,
            hidden_features=model_channels * hidden_multiplier,
            out_features=model_channels * 3,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )

        # Temporal encoding
        self.temporal_pos = nn.Parameter(torch.randn(num_frames, model_channels) * 0.02)

    def forward(
        self,
        dynamics: torch.Tensor,
        dynamics_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-frame dynamics embeddings.

        Args:
            dynamics: Agent dynamics [B, T, max_agents, state_dim]
            dynamics_mask: Optional mask for valid agents [B, T, max_agents]

        Returns:
            Tuple of:
                - t_emb: Per-frame embedding [B, T, model_channels]
                - adaln_emb: Per-frame AdaLN embedding [B, T, model_channels*3]
        """
        B, T, N, S = dynamics.shape

        if dynamics_mask is not None:
            dynamics = dynamics * dynamics_mask.unsqueeze(-1)

        # Process each frame
        dynamics_flat = rearrange(dynamics, 'b t n s -> (b t) (n s)')  # [B*T, N*S]

        t_emb = self.frame_embedder_D(dynamics_flat)  # [B*T, model_channels]
        adaln_emb = self.frame_embedder_3D(dynamics_flat)  # [B*T, model_channels*3]

        # Reshape back
        t_emb = rearrange(t_emb, '(b t) d -> b t d', b=B, t=T)
        adaln_emb = rearrange(adaln_emb, '(b t) d -> b t d', b=B, t=T)

        # Add temporal position encoding
        t_emb = t_emb + self.temporal_pos[:T]

        return t_emb, adaln_emb


class CounterfactualDynamicsEmbedder(nn.Module):
    """Specialized embedder for counterfactual scenario generation.

    This embedder is designed for specifying modifications to specific
    agents' dynamics for "what if" scenario generation.

    Example usage:
        - "What if car A moved 2 m/s faster?"
        - "What if the pedestrian crossed the street?"
    """

    def __init__(
        self,
        max_agents: int,
        state_dim: int,
        model_channels: int,
        hidden_multiplier: int = 4,
    ):
        """Initialize CounterfactualDynamicsEmbedder.

        Args:
            max_agents: Maximum number of agents
            state_dim: Dimension of per-agent state
            model_channels: DiT model channel dimension
            hidden_multiplier: Multiplier for hidden layer size
        """
        super().__init__()
        self.max_agents = max_agents
        self.state_dim = state_dim
        self.model_channels = model_channels

        # Per-agent embedding (allowing selective modification)
        self.agent_embedder = nn.Sequential(
            nn.Linear(state_dim, model_channels),
            nn.GELU(),
            nn.Linear(model_channels, model_channels),
        )

        # Modification embedder (encodes the delta)
        self.delta_embedder = nn.Sequential(
            nn.Linear(state_dim, model_channels),
            nn.GELU(),
            nn.Linear(model_channels, model_channels),
        )

        # Aggregation across agents
        self.aggregator_D = Mlp(
            in_features=model_channels * max_agents,
            hidden_features=model_channels * hidden_multiplier,
            out_features=model_channels,
        )

        self.aggregator_3D = Mlp(
            in_features=model_channels * max_agents,
            hidden_features=model_channels * hidden_multiplier,
            out_features=model_channels * 3,
        )

    def forward(
        self,
        base_dynamics: torch.Tensor,
        modified_dynamics: torch.Tensor,
        modification_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute counterfactual dynamics embeddings.

        Args:
            base_dynamics: Original dynamics [B, max_agents, state_dim]
            modified_dynamics: Modified dynamics [B, max_agents, state_dim]
            modification_mask: Mask indicating which agents are modified [B, max_agents]

        Returns:
            Tuple of:
                - t_emb: Embedding [B, 1, model_channels]
                - adaln_emb: AdaLN embedding [B, 1, model_channels*3]
        """
        B, N, S = base_dynamics.shape

        # Embed base dynamics
        base_emb = self.agent_embedder(base_dynamics)  # [B, N, model_channels]

        # Embed delta for modified agents
        delta = modified_dynamics - base_dynamics  # [B, N, S]
        delta_emb = self.delta_embedder(delta)  # [B, N, model_channels]

        # Combine: use delta_emb for modified agents, zero for others
        modification_mask = modification_mask.unsqueeze(-1)  # [B, N, 1]
        combined_emb = base_emb + delta_emb * modification_mask  # [B, N, model_channels]

        # Flatten and aggregate
        combined_flat = rearrange(combined_emb, 'b n d -> b (n d)')

        t_emb = self.aggregator_D(combined_flat).unsqueeze(1)  # [B, 1, model_channels]
        adaln_emb = self.aggregator_3D(combined_flat).unsqueeze(1)  # [B, 1, model_channels*3]

        return t_emb, adaln_emb
