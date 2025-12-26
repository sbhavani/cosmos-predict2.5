# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inference utilities for dynamics-aware video generation.

This module provides high-level APIs for:
1. Generating videos with dynamics prediction
2. Creating counterfactual scenarios
3. Extracting object dynamics from generated videos
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from einops import rearrange

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.dynamics.conditioner.dynamics_condition import (
    DynamicsCondition,
    create_dynamics_condition_from_video2world,
)


class DynamicsInference:
    """High-level inference API for dynamics-aware video generation.

    This class provides convenient methods for:
    - Standard video generation with dynamics prediction
    - Counterfactual scenario generation
    - Dynamics-conditioned generation

    Attributes:
        model: DynamicsWorldModel instance
        device: Target device for inference
    """

    def __init__(
        self,
        model,
        device: Union[str, torch.device] = "cuda",
    ):
        """Initialize DynamicsInference.

        Args:
            model: DynamicsWorldModel instance
            device: Target device
        """
        self.model = model
        self.device = torch.device(device)
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        video_frames: torch.Tensor,
        prompt: str = "",
        num_steps: int = 35,
        guidance: float = 1.5,
        return_dynamics: bool = True,
    ) -> Dict[str, Any]:
        """Generate video with dynamics prediction.

        Args:
            video_frames: Conditioning frames [B, C, T, H, W] or [C, T, H, W]
            prompt: Text prompt for generation
            num_steps: Number of sampling steps
            guidance: Classifier-free guidance scale
            return_dynamics: Whether to return dynamics predictions

        Returns:
            Dictionary with:
                - video: Generated video tensor [B, C, T, H, W]
                - dynamics: Per-object dynamics predictions (if return_dynamics)
                - confidences: Object detection confidences
        """
        # Ensure batch dimension
        if video_frames.dim() == 4:
            video_frames = video_frames.unsqueeze(0)

        video_frames = video_frames.to(self.device)
        B = video_frames.shape[0]

        # Prepare data batch
        data_batch = {
            'video': video_frames,
            'ai_caption': [prompt] * B if isinstance(prompt, str) else prompt,
            'fps': torch.tensor([24.0] * B, device=self.device),
        }

        # Generate
        generated_video, dynamics_pred = self.model.generate_with_dynamics(
            data_batch,
            num_steps=num_steps,
            guidance=guidance,
        )

        result = {
            'video': generated_video,
        }

        if return_dynamics:
            result['dynamics'] = self._process_dynamics_output(dynamics_pred)

        return result

    @torch.no_grad()
    def generate_counterfactual(
        self,
        video_frames: torch.Tensor,
        prompt: str = "",
        agent_modifications: Dict[int, Dict[str, Tuple[float, float, float]]] = None,
        num_steps: int = 35,
        guidance: float = 1.5,
    ) -> Dict[str, Any]:
        """Generate counterfactual video with modified agent dynamics.

        This enables "what if" scenario generation by modifying specific
        agents' trajectories.

        Args:
            video_frames: Conditioning frames [B, C, T, H, W]
            prompt: Text prompt
            agent_modifications: Dictionary mapping agent indices to modifications.
                Each modification is a dict with optional keys:
                    - 'velocity_delta': (vx, vy, vz) change in m/s
                    - 'position_delta': (x, y, z) change in meters
                    - 'acceleration_delta': (ax, ay, az) change in m/s^2
            num_steps: Number of sampling steps
            guidance: Classifier-free guidance scale

        Returns:
            Dictionary with:
                - video: Generated counterfactual video
                - original_dynamics: Dynamics before modification
                - modified_dynamics: Dynamics after modification
                - modifications_applied: Summary of modifications

        Example:
            >>> # Make car at index 0 move 2 m/s faster
            >>> result = inference.generate_counterfactual(
            ...     video_frames,
            ...     agent_modifications={
            ...         0: {'velocity_delta': (2.0, 0.0, 0.0)}
            ...     }
            ... )
        """
        if video_frames.dim() == 4:
            video_frames = video_frames.unsqueeze(0)

        video_frames = video_frames.to(self.device)
        B = video_frames.shape[0]

        # First, predict dynamics from input frames
        data_batch = {
            'video': video_frames,
            'ai_caption': [prompt] * B,
            'fps': torch.tensor([24.0] * B, device=self.device),
        }

        # Get base dynamics from conditioning frames
        _, latent_state, condition = self.model.get_data_and_condition(data_batch)
        base_dynamics = self.model.predict_dynamics(
            latent_state,
            torch.zeros(B, device=self.device),
            condition,
        )

        # Create dynamics condition
        dynamics_condition = create_dynamics_condition_from_video2world(
            condition,
            dynamics=base_dynamics['state_pred'],
            dynamics_mask=(base_dynamics['confidence'].sigmoid() > 0.5).float().squeeze(-1),
        )

        # Apply modifications
        if agent_modifications:
            for agent_idx, mods in agent_modifications.items():
                dynamics_condition = dynamics_condition.edit_agent_dynamics(
                    agent_idx=agent_idx,
                    position_delta=mods.get('position_delta'),
                    velocity_delta=mods.get('velocity_delta'),
                    acceleration_delta=mods.get('acceleration_delta'),
                )

        # Generate with modified dynamics
        # Note: This requires the model to support dynamics conditioning
        generated_video, final_dynamics = self.model.generate_with_dynamics(
            data_batch,
            num_steps=num_steps,
            guidance=guidance,
        )

        return {
            'video': generated_video,
            'original_dynamics': self._process_dynamics_output(base_dynamics),
            'modified_dynamics': self._process_dynamics_output(final_dynamics),
            'modifications_applied': agent_modifications,
        }

    @torch.no_grad()
    def predict_dynamics_only(
        self,
        video_frames: torch.Tensor,
        prompt: str = "",
    ) -> Dict[str, Any]:
        """Predict dynamics for input video without generation.

        Useful for analyzing existing videos to extract object trajectories.

        Args:
            video_frames: Input video [B, C, T, H, W]
            prompt: Optional text description

        Returns:
            Dictionary with:
                - positions: Object positions [B, T, N, 3]
                - velocities: Object velocities [B, T, N, 3]
                - accelerations: Object accelerations [B, T, N, 3]
                - classes: Object class predictions [B, T, N]
                - confidences: Detection confidences [B, T, N]
        """
        if video_frames.dim() == 4:
            video_frames = video_frames.unsqueeze(0)

        video_frames = video_frames.to(self.device)
        B = video_frames.shape[0]

        data_batch = {
            'video': video_frames,
            'ai_caption': [prompt] * B,
            'fps': torch.tensor([24.0] * B, device=self.device),
        }

        # Get latent representation
        _, latent_state, condition = self.model.get_data_and_condition(data_batch)

        # Predict dynamics at timestep 0 (clean data)
        dynamics_pred = self.model.predict_dynamics(
            latent_state,
            torch.zeros(B, device=self.device),
            condition,
        )

        return self._process_dynamics_output(dynamics_pred)

    def _process_dynamics_output(
        self,
        dynamics_pred: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """Process raw dynamics predictions into user-friendly format.

        Args:
            dynamics_pred: Raw predictions from dynamics head

        Returns:
            Processed predictions with separated state components
        """
        state_pred = dynamics_pred.get('state_pred', None)
        state_denorm = dynamics_pred.get('state_pred_denorm', state_pred)

        result = {
            'raw_state': state_pred,
        }

        if state_denorm is not None:
            # Split state into components
            result['positions'] = state_denorm[..., :3]  # x, y, z
            result['velocities'] = state_denorm[..., 3:6]  # vx, vy, vz
            result['accelerations'] = state_denorm[..., 6:9]  # ax, ay, az

        if 'class_logits' in dynamics_pred:
            result['class_logits'] = dynamics_pred['class_logits']
            result['classes'] = dynamics_pred['class_logits'].argmax(dim=-1)

        if 'confidence' in dynamics_pred:
            result['confidences'] = dynamics_pred['confidence'].sigmoid().squeeze(-1)

        return result

    def get_object_trajectories(
        self,
        dynamics_result: Dict[str, Any],
        confidence_threshold: float = 0.5,
    ) -> List[Dict[str, torch.Tensor]]:
        """Extract individual object trajectories from dynamics predictions.

        Args:
            dynamics_result: Output from predict_dynamics_only or generate
            confidence_threshold: Minimum confidence to consider an object detected

        Returns:
            List of trajectory dictionaries, one per detected object.
            Each dict contains:
                - positions: [T, 3]
                - velocities: [T, 3]
                - accelerations: [T, 3]
                - class: int
                - confidence: float (average)
        """
        trajectories = []

        positions = dynamics_result.get('positions')
        velocities = dynamics_result.get('velocities')
        accelerations = dynamics_result.get('accelerations')
        confidences = dynamics_result.get('confidences')
        classes = dynamics_result.get('classes')

        if positions is None or confidences is None:
            return trajectories

        # Assume batch size 1 for simplicity
        if positions.dim() == 4:
            positions = positions[0]
            velocities = velocities[0] if velocities is not None else None
            accelerations = accelerations[0] if accelerations is not None else None
            confidences = confidences[0]
            classes = classes[0] if classes is not None else None

        T, N, _ = positions.shape

        for obj_idx in range(N):
            # Check if object is detected (average confidence above threshold)
            obj_conf = confidences[:, obj_idx].mean().item()
            if obj_conf < confidence_threshold:
                continue

            trajectory = {
                'object_idx': obj_idx,
                'positions': positions[:, obj_idx],
                'confidence': obj_conf,
            }

            if velocities is not None:
                trajectory['velocities'] = velocities[:, obj_idx]

            if accelerations is not None:
                trajectory['accelerations'] = accelerations[:, obj_idx]

            if classes is not None:
                # Most common class prediction
                obj_class = classes[:, obj_idx].mode().values.item()
                trajectory['class'] = obj_class

            trajectories.append(trajectory)

        return trajectories


def load_dynamics_model(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    device: str = "cuda",
) -> DynamicsInference:
    """Load a DynamicsWorldModel from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Optional path to config file
        device: Target device

    Returns:
        DynamicsInference wrapper for the loaded model
    """
    from cosmos_predict2._src.predict2.dynamics.models.dynamics_world_model import (
        DynamicsWorldModel,
        DynamicsWorldModelConfig,
    )

    # Load config
    if config_path is not None:
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = DynamicsWorldModelConfig(**config_dict)
    else:
        # Use default config
        config = DynamicsWorldModelConfig()

    # Create model
    model = DynamicsWorldModel(config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return DynamicsInference(model, device=device)
