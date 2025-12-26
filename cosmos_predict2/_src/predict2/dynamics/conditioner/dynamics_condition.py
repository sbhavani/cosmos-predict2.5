# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamics condition for counterfactual video generation.

This module extends the Video2World condition to support dynamics-based
conditioning, enabling:
1. Conditioning on observed object dynamics
2. Counterfactual editing of specific object trajectories
"""

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Tuple, Union

import torch

from cosmos_predict2._src.predict2.conditioner import DataType, Text2WorldCondition
from cosmos_predict2._src.predict2.configs.video2world.defaults.conditioner import Video2WorldCondition


@dataclass(frozen=True)
class DynamicsCondition(Video2WorldCondition):
    """Condition with per-object dynamics for counterfactual generation.

    Extends Video2WorldCondition to include dynamics information that can
    be used to condition video generation on specific object trajectories.

    Attributes:
        target_dynamics: Target dynamics to condition on [B, T, max_agents, state_dim]
        dynamics_mask: Mask indicating which agents have dynamics [B, T, max_agents]
        modification_mask: Mask for agents with modified dynamics [B, max_agents]
        base_dynamics: Original dynamics before modification [B, T, max_agents, state_dim]
    """
    target_dynamics: Optional[torch.Tensor] = None
    dynamics_mask: Optional[torch.Tensor] = None
    modification_mask: Optional[torch.Tensor] = None
    base_dynamics: Optional[torch.Tensor] = None

    @property
    def has_dynamics(self) -> bool:
        """Check if dynamics conditioning is available."""
        return self.target_dynamics is not None

    @property
    def is_counterfactual(self) -> bool:
        """Check if this is a counterfactual scenario."""
        return self.modification_mask is not None and self.modification_mask.any()

    def set_dynamics_condition(
        self,
        dynamics: torch.Tensor,
        dynamics_mask: Optional[torch.Tensor] = None,
    ) -> "DynamicsCondition":
        """Set dynamics for conditioning.

        Args:
            dynamics: Dynamics tensor [B, T, N, state_dim]
            dynamics_mask: Optional mask for valid agents [B, T, N]

        Returns:
            New DynamicsCondition with dynamics set
        """
        kwargs = self.to_dict(skip_underscore=False)
        kwargs['target_dynamics'] = dynamics.detach()
        if dynamics_mask is not None:
            kwargs['dynamics_mask'] = dynamics_mask.detach()
        else:
            # Create mask based on non-zero dynamics
            kwargs['dynamics_mask'] = (dynamics.abs().sum(dim=-1) > 0).float()
        return type(self)(**kwargs)

    def edit_agent_dynamics(
        self,
        agent_idx: int,
        position_delta: Optional[Tuple[float, float, float]] = None,
        velocity_delta: Optional[Tuple[float, float, float]] = None,
        acceleration_delta: Optional[Tuple[float, float, float]] = None,
    ) -> "DynamicsCondition":
        """Edit a specific agent's dynamics for counterfactual generation.

        This is the main interface for creating "what if" scenarios like
        "What if car A moved 2 m/s faster?"

        Args:
            agent_idx: Index of the agent to modify
            position_delta: Change in position (dx, dy, dz) in meters
            velocity_delta: Change in velocity (dvx, dvy, dvz) in m/s
            acceleration_delta: Change in acceleration (dax, day, daz) in m/s^2

        Returns:
            New DynamicsCondition with modified dynamics

        Example:
            # Make agent 0 move 2 m/s faster in x direction
            new_condition = condition.edit_agent_dynamics(
                agent_idx=0,
                velocity_delta=(2.0, 0.0, 0.0)
            )
        """
        if self.target_dynamics is None:
            raise ValueError("Cannot edit dynamics - no target_dynamics set")

        kwargs = self.to_dict(skip_underscore=False)

        # Clone dynamics for modification
        new_dynamics = self.target_dynamics.clone()
        B, T, N, S = new_dynamics.shape

        if agent_idx >= N:
            raise ValueError(f"Agent index {agent_idx} out of range (max {N-1})")

        # Apply position delta (dims 0:3)
        if position_delta is not None:
            delta = torch.tensor(position_delta, device=new_dynamics.device, dtype=new_dynamics.dtype)
            new_dynamics[:, :, agent_idx, :3] += delta

        # Apply velocity delta (dims 3:6)
        if velocity_delta is not None:
            delta = torch.tensor(velocity_delta, device=new_dynamics.device, dtype=new_dynamics.dtype)
            new_dynamics[:, :, agent_idx, 3:6] += delta

        # Apply acceleration delta (dims 6:9)
        if acceleration_delta is not None:
            delta = torch.tensor(acceleration_delta, device=new_dynamics.device, dtype=new_dynamics.dtype)
            new_dynamics[:, :, agent_idx, 6:9] += delta

        kwargs['target_dynamics'] = new_dynamics

        # Store base dynamics if not already set
        if self.base_dynamics is None:
            kwargs['base_dynamics'] = self.target_dynamics.clone()

        # Update modification mask
        if self.modification_mask is None:
            modification_mask = torch.zeros(B, N, device=new_dynamics.device, dtype=torch.float32)
        else:
            modification_mask = self.modification_mask.clone()
        modification_mask[:, agent_idx] = 1.0
        kwargs['modification_mask'] = modification_mask

        return type(self)(**kwargs)

    def edit_agent_trajectory(
        self,
        agent_idx: int,
        new_positions: torch.Tensor,
    ) -> "DynamicsCondition":
        """Replace an agent's entire trajectory.

        Args:
            agent_idx: Index of the agent to modify
            new_positions: New position trajectory [B, T, 3] or [T, 3]

        Returns:
            New DynamicsCondition with modified trajectory
        """
        if self.target_dynamics is None:
            raise ValueError("Cannot edit dynamics - no target_dynamics set")

        kwargs = self.to_dict(skip_underscore=False)

        new_dynamics = self.target_dynamics.clone()
        B, T, N, S = new_dynamics.shape

        # Handle input shape
        if new_positions.dim() == 2:
            new_positions = new_positions.unsqueeze(0).expand(B, -1, -1)

        # Replace positions
        new_dynamics[:, :, agent_idx, :3] = new_positions

        # Recompute velocities from new positions (central difference)
        # v[t] = (p[t+1] - p[t-1]) / (2 * dt)
        # For simplicity, use forward difference: v[t] = (p[t+1] - p[t]) / dt
        # Assume dt = 1 for normalized time
        if T > 1:
            velocities = torch.zeros_like(new_positions)
            velocities[:, :-1] = new_positions[:, 1:] - new_positions[:, :-1]
            velocities[:, -1] = velocities[:, -2]  # Extrapolate last frame
            new_dynamics[:, :, agent_idx, 3:6] = velocities

        # Recompute accelerations
        if T > 2:
            accelerations = torch.zeros_like(new_positions)
            accelerations[:, :-1] = velocities[:, 1:] - velocities[:, :-1]
            accelerations[:, -1] = accelerations[:, -2]
            new_dynamics[:, :, agent_idx, 6:9] = accelerations

        kwargs['target_dynamics'] = new_dynamics

        # Store base and update mask
        if self.base_dynamics is None:
            kwargs['base_dynamics'] = self.target_dynamics.clone()

        if self.modification_mask is None:
            modification_mask = torch.zeros(B, N, device=new_dynamics.device, dtype=torch.float32)
        else:
            modification_mask = self.modification_mask.clone()
        modification_mask[:, agent_idx] = 1.0
        kwargs['modification_mask'] = modification_mask

        return type(self)(**kwargs)

    def clear_modifications(self) -> "DynamicsCondition":
        """Reset dynamics to original (before modifications).

        Returns:
            New DynamicsCondition with original dynamics restored
        """
        if self.base_dynamics is None:
            return self

        kwargs = self.to_dict(skip_underscore=False)
        kwargs['target_dynamics'] = self.base_dynamics.clone()
        kwargs['modification_mask'] = None
        kwargs['base_dynamics'] = None
        return type(self)(**kwargs)

    def get_dynamics_embedding_inputs(self) -> Dict[str, torch.Tensor]:
        """Get inputs for the dynamics embedder.

        Returns:
            Dictionary with tensors needed for dynamics embedding
        """
        inputs = {}
        if self.target_dynamics is not None:
            inputs['dynamics'] = self.target_dynamics
        if self.dynamics_mask is not None:
            inputs['dynamics_mask'] = self.dynamics_mask
        if self.is_counterfactual:
            inputs['base_dynamics'] = self.base_dynamics
            inputs['modification_mask'] = self.modification_mask
        return inputs


def create_dynamics_condition_from_video2world(
    video2world_condition: Video2WorldCondition,
    dynamics: Optional[torch.Tensor] = None,
    dynamics_mask: Optional[torch.Tensor] = None,
) -> DynamicsCondition:
    """Create a DynamicsCondition from an existing Video2WorldCondition.

    Args:
        video2world_condition: Base video condition
        dynamics: Optional dynamics tensor
        dynamics_mask: Optional dynamics mask

    Returns:
        New DynamicsCondition with all properties from base plus dynamics
    """
    # Get all fields from the base condition
    kwargs = video2world_condition.to_dict(skip_underscore=False)

    # Add dynamics fields
    kwargs['target_dynamics'] = dynamics
    kwargs['dynamics_mask'] = dynamics_mask
    kwargs['modification_mask'] = None
    kwargs['base_dynamics'] = None

    return DynamicsCondition(**kwargs)
