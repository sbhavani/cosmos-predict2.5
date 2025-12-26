# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hungarian matching for DETR-style object detection training.

This module implements the Hungarian algorithm to optimally assign predicted
objects to ground truth objects based on a cost matrix that considers
class, position, and velocity similarity.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    """Hungarian matcher for assigning predictions to ground truth.

    The matcher computes a cost matrix between predictions and targets,
    then uses the Hungarian algorithm to find the optimal assignment.

    The cost is a weighted combination of:
    - Classification cost: negative class probability
    - Position cost: L1 distance between predicted and target positions
    - Velocity cost: L2 distance between predicted and target velocities

    Attributes:
        class_cost: Weight for classification cost
        position_cost: Weight for position cost
        velocity_cost: Weight for velocity cost
    """

    def __init__(
        self,
        class_cost: float = 1.0,
        position_cost: float = 5.0,
        velocity_cost: float = 2.0,
    ):
        """Initialize HungarianMatcher.

        Args:
            class_cost: Weight for classification cost
            position_cost: Weight for position cost
            velocity_cost: Weight for velocity cost
        """
        super().__init__()
        self.class_cost = class_cost
        self.position_cost = position_cost
        self.velocity_cost = velocity_cost

    @torch.no_grad()
    def forward(
        self,
        pred_class_logits: torch.Tensor,
        pred_states: torch.Tensor,
        target_classes: torch.Tensor,
        target_states: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> List[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Compute optimal assignment between predictions and targets.

        Args:
            pred_class_logits: Predicted class logits [B, T, N, num_classes+1]
            pred_states: Predicted states [B, T, N, state_dim]
            target_classes: Target class indices [B, T, M]
            target_states: Target states [B, T, M, state_dim]
            valid_mask: Boolean mask for valid targets [B, T, M]

        Returns:
            List of lists (per batch, per frame) of tuples (pred_indices, target_indices)
            where each tuple contains tensors of matched indices.
        """
        B, T, N, C = pred_class_logits.shape
        M = target_classes.shape[2]

        # Convert logits to probabilities
        pred_probs = pred_class_logits.softmax(dim=-1)  # [B, T, N, C]

        all_indices = []

        for b in range(B):
            batch_indices = []
            for t in range(T):
                # Get predictions for this (batch, frame)
                pred_prob = pred_probs[b, t]  # [N, C]
                pred_state = pred_states[b, t]  # [N, state_dim]

                # Get targets for this (batch, frame)
                tgt_class = target_classes[b, t]  # [M]
                tgt_state = target_states[b, t]  # [M, state_dim]
                mask = valid_mask[b, t]  # [M]

                # Count valid targets
                num_valid = int(mask.sum().item())

                if num_valid == 0:
                    # No valid targets, return empty indices
                    batch_indices.append((
                        torch.tensor([], dtype=torch.long, device=pred_class_logits.device),
                        torch.tensor([], dtype=torch.long, device=pred_class_logits.device),
                    ))
                    continue

                # Get valid targets only
                valid_tgt_class = tgt_class[:num_valid]  # [num_valid]
                valid_tgt_state = tgt_state[:num_valid]  # [num_valid, state_dim]

                # Compute classification cost
                # For each prediction, get the probability of the target class
                # Higher probability = lower cost
                class_cost = -pred_prob[:, valid_tgt_class]  # [N, num_valid]

                # Compute position cost (L1 distance)
                # Position is in dims 0:3 (x, y, z)
                pred_pos = pred_state[:, :3]  # [N, 3]
                tgt_pos = valid_tgt_state[:, :3]  # [num_valid, 3]
                position_cost = torch.cdist(pred_pos, tgt_pos, p=1)  # [N, num_valid]

                # Compute velocity cost (L2 distance)
                # Velocity is in dims 3:6 (vx, vy, vz)
                pred_vel = pred_state[:, 3:6]  # [N, 3]
                tgt_vel = valid_tgt_state[:, 3:6]  # [num_valid, 3]
                velocity_cost = torch.cdist(pred_vel, tgt_vel, p=2)  # [N, num_valid]

                # Combined cost matrix
                cost_matrix = (
                    self.class_cost * class_cost +
                    self.position_cost * position_cost +
                    self.velocity_cost * velocity_cost
                )

                # Run Hungarian algorithm
                cost_np = cost_matrix.cpu().numpy()
                pred_idx, tgt_idx = linear_sum_assignment(cost_np)

                # Convert to tensors
                pred_idx = torch.tensor(pred_idx, dtype=torch.long, device=pred_class_logits.device)
                tgt_idx = torch.tensor(tgt_idx, dtype=torch.long, device=pred_class_logits.device)

                batch_indices.append((pred_idx, tgt_idx))

            all_indices.append(batch_indices)

        return all_indices


class HungarianMatcherBatched(nn.Module):
    """Batched Hungarian matcher for better GPU utilization.

    This version processes all frames in a batch together for efficiency
    when the number of valid targets is similar across frames.
    """

    def __init__(
        self,
        class_cost: float = 1.0,
        position_cost: float = 5.0,
        velocity_cost: float = 2.0,
    ):
        super().__init__()
        self.matcher = HungarianMatcher(class_cost, position_cost, velocity_cost)

    @torch.no_grad()
    def forward(
        self,
        outputs: dict,
        targets: dict,
    ) -> List[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Compute optimal assignment.

        Args:
            outputs: Dictionary with 'class_logits' and 'state_pred'
            targets: Dictionary with 'classes', 'states', and 'valid_mask'

        Returns:
            Matched indices per (batch, frame)
        """
        return self.matcher(
            outputs['class_logits'],
            outputs['state_pred'],
            targets['classes'],
            targets['states'],
            targets['valid_mask'],
        )
