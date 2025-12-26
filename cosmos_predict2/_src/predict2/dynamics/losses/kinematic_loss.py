# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kinematic constraint loss for physics-consistent dynamics prediction.

This module implements loss terms that enforce Newtonian physics constraints:
- Position consistency: p_{t+1} = p_t + v_t * dt + 0.5 * a_t * dt^2
- Velocity consistency: v_{t+1} = v_t + a_t * dt

These constraints help the model learn physically plausible dynamics.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class KinematicConstraintLoss(nn.Module):
    """Loss for enforcing kinematic consistency.

    Enforces that predicted positions, velocities, and accelerations
    are consistent with Newtonian physics equations of motion.

    Attributes:
        position_weight: Weight for position consistency loss
        velocity_weight: Weight for velocity consistency loss
        use_acceleration: Whether to use acceleration in position update
    """

    def __init__(
        self,
        position_weight: float = 1.0,
        velocity_weight: float = 0.5,
        use_acceleration: bool = True,
    ):
        """Initialize KinematicConstraintLoss.

        Args:
            position_weight: Weight for position consistency term
            velocity_weight: Weight for velocity consistency term
            use_acceleration: If True, use full kinematic equation with acceleration
        """
        super().__init__()
        self.position_weight = position_weight
        self.velocity_weight = velocity_weight
        self.use_acceleration = use_acceleration

    def forward(
        self,
        pred_states: torch.Tensor,
        dt: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute kinematic constraint loss.

        Args:
            pred_states: Predicted states [B, T, N, 9] where 9 = (x,y,z,vx,vy,vz,ax,ay,az)
            dt: Time deltas between frames [B, T-1]
            valid_mask: Boolean mask for valid agents [B, T, N]

        Returns:
            Dictionary with:
                - kinematic_loss: Total kinematic constraint loss
                - position_consistency_loss: Position consistency term
                - velocity_consistency_loss: Velocity consistency term
        """
        B, T, N, S = pred_states.shape

        # Ensure we have enough frames for temporal constraints
        if T < 2:
            zero = torch.tensor(0.0, device=pred_states.device, dtype=pred_states.dtype)
            return {
                'kinematic_loss': zero,
                'position_consistency_loss': zero,
                'velocity_consistency_loss': zero,
            }

        # Extract position, velocity, acceleration
        pos = pred_states[..., :3]   # [B, T, N, 3] - x, y, z
        vel = pred_states[..., 3:6]  # [B, T, N, 3] - vx, vy, vz
        acc = pred_states[..., 6:9]  # [B, T, N, 3] - ax, ay, az

        # Expand dt for broadcasting: [B, T-1] -> [B, T-1, 1, 1]
        dt_expanded = dt[:, :, None, None]

        # Position consistency: p_{t+1} = p_t + v_t * dt + 0.5 * a_t * dt^2
        if self.use_acceleration:
            pos_predicted = (
                pos[:, :-1] +
                vel[:, :-1] * dt_expanded +
                0.5 * acc[:, :-1] * dt_expanded ** 2
            )
        else:
            pos_predicted = pos[:, :-1] + vel[:, :-1] * dt_expanded

        pos_target = pos[:, 1:]  # [B, T-1, N, 3]

        # Velocity consistency: v_{t+1} = v_t + a_t * dt
        vel_predicted = vel[:, :-1] + acc[:, :-1] * dt_expanded
        vel_target = vel[:, 1:]  # [B, T-1, N, 3]

        # Create temporal valid mask (both t and t+1 must be valid)
        temporal_valid = valid_mask[:, :-1] * valid_mask[:, 1:]  # [B, T-1, N]
        temporal_valid_expanded = temporal_valid[:, :, :, None]  # [B, T-1, N, 1]

        # Count valid pairs
        num_valid = temporal_valid.sum() + 1e-6

        # Compute MSE loss with masking
        pos_diff = (pos_predicted - pos_target) ** 2  # [B, T-1, N, 3]
        pos_loss = (pos_diff * temporal_valid_expanded).sum() / num_valid / 3

        vel_diff = (vel_predicted - vel_target) ** 2  # [B, T-1, N, 3]
        vel_loss = (vel_diff * temporal_valid_expanded).sum() / num_valid / 3

        # Total kinematic loss
        total_loss = self.position_weight * pos_loss + self.velocity_weight * vel_loss

        return {
            'kinematic_loss': total_loss,
            'position_consistency_loss': pos_loss,
            'velocity_consistency_loss': vel_loss,
        }


class SmoothMotionLoss(nn.Module):
    """Loss for encouraging smooth motion predictions.

    Penalizes sudden changes in acceleration (jerk) to encourage
    physically plausible, smooth trajectories.
    """

    def __init__(self, jerk_weight: float = 0.1):
        """Initialize SmoothMotionLoss.

        Args:
            jerk_weight: Weight for jerk penalty
        """
        super().__init__()
        self.jerk_weight = jerk_weight

    def forward(
        self,
        pred_states: torch.Tensor,
        dt: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute smooth motion loss.

        Args:
            pred_states: Predicted states [B, T, N, 9]
            dt: Time deltas [B, T-1]
            valid_mask: Valid agent mask [B, T, N]

        Returns:
            Dictionary with jerk_loss
        """
        B, T, N, S = pred_states.shape

        if T < 3:
            zero = torch.tensor(0.0, device=pred_states.device, dtype=pred_states.dtype)
            return {'jerk_loss': zero}

        # Extract acceleration
        acc = pred_states[..., 6:9]  # [B, T, N, 3]

        # Compute jerk (change in acceleration)
        # dt for jerk should average consecutive dt values
        dt_avg = (dt[:, :-1] + dt[:, 1:]) / 2  # [B, T-2]
        dt_avg = dt_avg[:, :, None, None]

        jerk = (acc[:, 1:-1] - acc[:, :-2]) / (dt_avg + 1e-6)  # [B, T-2, N, 3]

        # Valid mask for jerk (three consecutive frames must be valid)
        jerk_valid = valid_mask[:, :-2] * valid_mask[:, 1:-1] * valid_mask[:, 2:]
        jerk_valid = jerk_valid[:, :, :, None]

        num_valid = jerk_valid.sum() + 1e-6

        # Penalize large jerk values
        jerk_magnitude = (jerk ** 2 * jerk_valid).sum() / num_valid / 3

        return {
            'jerk_loss': self.jerk_weight * jerk_magnitude,
        }


class CollisionAvoidanceLoss(nn.Module):
    """Loss for encouraging collision-free predictions.

    Penalizes predictions where different objects occupy
    overlapping positions.
    """

    def __init__(self, min_distance: float = 1.0, collision_weight: float = 1.0):
        """Initialize CollisionAvoidanceLoss.

        Args:
            min_distance: Minimum allowed distance between objects (meters)
            collision_weight: Weight for collision penalty
        """
        super().__init__()
        self.min_distance = min_distance
        self.collision_weight = collision_weight

    def forward(
        self,
        pred_states: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute collision avoidance loss.

        Args:
            pred_states: Predicted states [B, T, N, 9]
            valid_mask: Valid agent mask [B, T, N]

        Returns:
            Dictionary with collision_loss
        """
        B, T, N, S = pred_states.shape

        # Extract positions
        pos = pred_states[..., :3]  # [B, T, N, 3]

        # Compute pairwise distances
        # pos_i: [B, T, N, 1, 3], pos_j: [B, T, 1, N, 3]
        pos_i = pos[:, :, :, None, :]
        pos_j = pos[:, :, None, :, :]
        distances = torch.sqrt(((pos_i - pos_j) ** 2).sum(dim=-1) + 1e-6)  # [B, T, N, N]

        # Create mask for valid pairs (both objects valid, not same object)
        valid_i = valid_mask[:, :, :, None]  # [B, T, N, 1]
        valid_j = valid_mask[:, :, None, :]  # [B, T, 1, N]
        pair_valid = valid_i * valid_j  # [B, T, N, N]

        # Exclude self-distances
        eye = torch.eye(N, device=pred_states.device, dtype=torch.bool)
        eye = eye[None, None, :, :]  # [1, 1, N, N]
        pair_valid = pair_valid * (~eye).float()

        # Compute collision penalty (hinge loss)
        collision_margin = F.relu(self.min_distance - distances)  # [B, T, N, N]
        collision_loss = (collision_margin * pair_valid).sum() / (pair_valid.sum() + 1e-6)

        return {
            'collision_loss': self.collision_weight * collision_loss,
        }
