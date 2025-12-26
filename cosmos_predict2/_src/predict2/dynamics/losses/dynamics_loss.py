# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Combined loss for dynamics prediction training.

This module combines multiple loss components for training the dynamics head:
- State prediction loss (MSE on matched position/velocity/acceleration)
- Classification loss (cross-entropy on object classes)
- Kinematic constraint loss (physics consistency)
- Optional collision avoidance loss
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cosmos_predict2._src.predict2.dynamics.losses.hungarian_matcher import HungarianMatcher
from cosmos_predict2._src.predict2.dynamics.losses.kinematic_loss import (
    CollisionAvoidanceLoss,
    KinematicConstraintLoss,
    SmoothMotionLoss,
)


class DynamicsLoss(nn.Module):
    """Combined loss for dynamics prediction.

    This loss function includes:
    1. State prediction loss: MSE between predicted and target states
    2. Classification loss: Cross-entropy for object class prediction
    3. Objectness loss: BCE for object presence prediction
    4. Kinematic constraint loss: Physics consistency regularization
    5. Optional auxiliary losses (collision, smoothness)

    The matching between predictions and targets is done using
    the Hungarian algorithm.

    Attributes:
        state_weight: Weight for state prediction loss
        class_weight: Weight for classification loss
        objectness_weight: Weight for objectness prediction loss
        kinematic_weight: Weight for kinematic constraints
        no_object_weight: Down-weight factor for no-object class
    """

    def __init__(
        self,
        state_weight: float = 1.0,
        class_weight: float = 1.0,
        objectness_weight: float = 1.0,
        kinematic_weight: float = 0.1,
        no_object_weight: float = 0.1,
        collision_weight: float = 0.0,
        smoothness_weight: float = 0.0,
        matcher_class_cost: float = 1.0,
        matcher_position_cost: float = 5.0,
        matcher_velocity_cost: float = 2.0,
    ):
        """Initialize DynamicsLoss.

        Args:
            state_weight: Weight for state prediction loss
            class_weight: Weight for classification loss
            objectness_weight: Weight for objectness loss
            kinematic_weight: Weight for kinematic constraints
            no_object_weight: Weight factor for no-object class in CE loss
            collision_weight: Weight for collision avoidance (0 to disable)
            smoothness_weight: Weight for smooth motion (0 to disable)
            matcher_class_cost: Hungarian matcher class cost
            matcher_position_cost: Hungarian matcher position cost
            matcher_velocity_cost: Hungarian matcher velocity cost
        """
        super().__init__()

        self.state_weight = state_weight
        self.class_weight = class_weight
        self.objectness_weight = objectness_weight
        self.kinematic_weight = kinematic_weight
        self.no_object_weight = no_object_weight

        # Hungarian matcher
        self.matcher = HungarianMatcher(
            class_cost=matcher_class_cost,
            position_cost=matcher_position_cost,
            velocity_cost=matcher_velocity_cost,
        )

        # Kinematic constraint loss
        self.kinematic_loss_fn = KinematicConstraintLoss()

        # Optional auxiliary losses
        self.collision_loss_fn = None
        self.smooth_loss_fn = None

        if collision_weight > 0:
            self.collision_loss_fn = CollisionAvoidanceLoss(collision_weight=collision_weight)

        if smoothness_weight > 0:
            self.smooth_loss_fn = SmoothMotionLoss(jerk_weight=smoothness_weight)

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        dt: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined dynamics loss.

        Args:
            predictions: Dictionary with:
                - class_logits: [B, T, N, num_classes+1]
                - state_pred: [B, T, N, state_dim]
                - confidence: [B, T, N, 1]
            targets: Dictionary with:
                - classes: [B, T, M] class indices
                - states: [B, T, M, state_dim] ground truth states
                - valid_mask: [B, T, M] boolean mask for valid targets
            dt: Time deltas between frames [B, T-1]

        Returns:
            Dictionary with all loss components and total loss
        """
        # Get dimensions
        B, T, N, C = predictions['class_logits'].shape
        num_classes = C - 1  # Exclude no-object class
        device = predictions['class_logits'].device

        # Run Hungarian matching
        indices = self.matcher(
            predictions['class_logits'],
            predictions['state_pred'],
            targets['classes'],
            targets['states'],
            targets['valid_mask'],
        )

        # Compute state loss on matched pairs
        state_loss = self._compute_state_loss(
            predictions['state_pred'],
            targets['states'],
            targets['valid_mask'],
            indices,
        )

        # Compute classification loss
        class_loss = self._compute_class_loss(
            predictions['class_logits'],
            targets['classes'],
            targets['valid_mask'],
            indices,
            num_classes,
        )

        # Compute objectness loss
        objectness_loss = self._compute_objectness_loss(
            predictions['confidence'],
            targets['valid_mask'],
            indices,
        )

        # Compute kinematic constraint loss
        kinematic_losses = self.kinematic_loss_fn(
            predictions['state_pred'],
            dt,
            self._get_predicted_valid_mask(predictions['confidence']),
        )

        # Total loss
        total_loss = (
            self.state_weight * state_loss +
            self.class_weight * class_loss +
            self.objectness_weight * objectness_loss +
            self.kinematic_weight * kinematic_losses['kinematic_loss']
        )

        # Build output dictionary
        output = {
            'total_dynamics_loss': total_loss,
            'state_loss': state_loss,
            'class_loss': class_loss,
            'objectness_loss': objectness_loss,
            **kinematic_losses,
        }

        # Optional auxiliary losses
        if self.collision_loss_fn is not None:
            collision_losses = self.collision_loss_fn(
                predictions['state_pred'],
                self._get_predicted_valid_mask(predictions['confidence']),
            )
            output.update(collision_losses)
            total_loss = total_loss + collision_losses['collision_loss']

        if self.smooth_loss_fn is not None:
            smooth_losses = self.smooth_loss_fn(
                predictions['state_pred'],
                dt,
                self._get_predicted_valid_mask(predictions['confidence']),
            )
            output.update(smooth_losses)
            total_loss = total_loss + smooth_losses['jerk_loss']

        output['total_dynamics_loss'] = total_loss

        return output

    def _compute_state_loss(
        self,
        pred_states: torch.Tensor,
        target_states: torch.Tensor,
        valid_mask: torch.Tensor,
        indices: List[List[Tuple[torch.Tensor, torch.Tensor]]],
    ) -> torch.Tensor:
        """Compute MSE loss on matched state predictions.

        Args:
            pred_states: [B, T, N, state_dim]
            target_states: [B, T, M, state_dim]
            valid_mask: [B, T, M]
            indices: Matched indices from Hungarian algorithm

        Returns:
            State prediction loss scalar
        """
        B, T, N, S = pred_states.shape
        device = pred_states.device

        total_loss = torch.tensor(0.0, device=device)
        num_matched = 0

        for b in range(B):
            for t in range(T):
                pred_idx, tgt_idx = indices[b][t]

                if len(pred_idx) == 0:
                    continue

                # Get matched predictions and targets
                matched_pred = pred_states[b, t, pred_idx]  # [num_matched, S]
                matched_tgt = target_states[b, t, tgt_idx]  # [num_matched, S]

                # MSE loss
                loss = F.mse_loss(matched_pred, matched_tgt, reduction='sum')
                total_loss = total_loss + loss
                num_matched += len(pred_idx) * S

        return total_loss / (num_matched + 1e-6)

    def _compute_class_loss(
        self,
        pred_logits: torch.Tensor,
        target_classes: torch.Tensor,
        valid_mask: torch.Tensor,
        indices: List[List[Tuple[torch.Tensor, torch.Tensor]]],
        num_classes: int,
    ) -> torch.Tensor:
        """Compute classification loss.

        For matched predictions, compute CE with target class.
        For unmatched predictions, they should predict no-object.

        Args:
            pred_logits: [B, T, N, num_classes+1]
            target_classes: [B, T, M]
            valid_mask: [B, T, M]
            indices: Matched indices
            num_classes: Number of object classes (excluding no-object)

        Returns:
            Classification loss scalar
        """
        B, T, N, C = pred_logits.shape
        device = pred_logits.device

        # Create target tensor: default to no-object class (last class)
        target_labels = torch.full((B, T, N), num_classes, dtype=torch.long, device=device)

        # Fill in matched targets
        for b in range(B):
            for t in range(T):
                pred_idx, tgt_idx = indices[b][t]
                if len(pred_idx) > 0:
                    target_labels[b, t, pred_idx] = target_classes[b, t, tgt_idx]

        # Create class weights (down-weight no-object class)
        class_weights = torch.ones(C, device=device)
        class_weights[-1] = self.no_object_weight

        # Flatten and compute CE loss
        pred_flat = pred_logits.view(-1, C)  # [B*T*N, C]
        target_flat = target_labels.view(-1)  # [B*T*N]

        loss = F.cross_entropy(pred_flat, target_flat, weight=class_weights)

        return loss

    def _compute_objectness_loss(
        self,
        pred_confidence: torch.Tensor,
        valid_mask: torch.Tensor,
        indices: List[List[Tuple[torch.Tensor, torch.Tensor]]],
    ) -> torch.Tensor:
        """Compute objectness (object presence) loss.

        Matched predictions should have high confidence,
        unmatched predictions should have low confidence.

        Args:
            pred_confidence: [B, T, N, 1]
            valid_mask: [B, T, M]
            indices: Matched indices

        Returns:
            Objectness loss scalar
        """
        B, T, N, _ = pred_confidence.shape
        device = pred_confidence.device

        # Create target: 1 for matched, 0 for unmatched
        target_objectness = torch.zeros(B, T, N, device=device)

        for b in range(B):
            for t in range(T):
                pred_idx, _ = indices[b][t]
                if len(pred_idx) > 0:
                    target_objectness[b, t, pred_idx] = 1.0

        # BCE loss
        pred_flat = pred_confidence.squeeze(-1).view(-1)  # [B*T*N]
        target_flat = target_objectness.view(-1)  # [B*T*N]

        loss = F.binary_cross_entropy_with_logits(pred_flat, target_flat)

        return loss

    def _get_predicted_valid_mask(self, confidence: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
        """Create valid mask from predicted confidence scores.

        Args:
            confidence: [B, T, N, 1] confidence logits
            threshold: Threshold for considering an object present

        Returns:
            Valid mask [B, T, N]
        """
        return (confidence.squeeze(-1).sigmoid() > threshold).float()
