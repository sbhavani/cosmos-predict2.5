# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamics-aware world model for joint video and dynamics prediction.

This module extends the Video2World model to jointly predict:
1. Future video frames (original objective)
2. Per-object dynamics (position, velocity, acceleration)

The dynamics prediction uses a DETR-style detection head that extracts
features from the DiT backbone and predicts per-object states.
"""

from typing import Any, Dict, List, Optional, Tuple

import attrs
import torch
from einops import rearrange
from torch import Tensor

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.models.video2world_model_rectified_flow import (
    Video2WorldModelRectifiedFlow,
    Video2WorldModelRectifiedFlowConfig,
)
from cosmos_predict2._src.predict2.dynamics.modules.dynamics_head import (
    DynamicsHead,
    DynamicsHeadConfig,
)
from cosmos_predict2._src.predict2.dynamics.modules.dynamics_embedder import (
    DynamicsEmbedder,
)
from cosmos_predict2._src.predict2.dynamics.losses.dynamics_loss import DynamicsLoss


@attrs.define(slots=False)
class DynamicsWorldModelConfig(Video2WorldModelRectifiedFlowConfig):
    """Configuration for DynamicsWorldModel.

    Extends Video2World config with dynamics-specific parameters.

    Attributes:
        num_object_queries: Number of object queries for DETR-style detection
        dynamics_hidden_dim: Hidden dimension for dynamics head
        num_decoder_layers: Number of transformer decoder layers
        intermediate_feature_ids: Which backbone layers to extract features from
        video_loss_weight: Weight for video prediction loss
        dynamics_loss_weight: Weight for dynamics prediction loss
        kinematic_loss_weight: Weight for kinematic constraints
        use_dynamics_conditioning: Whether to condition on dynamics
        max_agents: Maximum number of agents
        state_dim: Dimension of per-agent state
        num_classes: Number of object classes
        dynamics_mean: Normalization mean for dynamics
        dynamics_std: Normalization std for dynamics
    """
    # Dynamics head configuration
    num_object_queries: int = 32
    dynamics_hidden_dim: int = 256
    num_decoder_layers: int = 6
    num_attention_heads: int = 8
    dynamics_feedforward_dim: int = 1024
    dynamics_dropout: float = 0.1

    # Feature extraction
    intermediate_feature_ids: Optional[List[int]] = None  # Will be auto-computed if None

    # Loss weights
    video_loss_weight: float = 1.0
    dynamics_loss_weight: float = 0.1
    kinematic_loss_weight: float = 0.01

    # Dynamics conditioning
    use_dynamics_conditioning: bool = False

    # Agent configuration
    max_agents: int = 32
    state_dim: int = 9
    num_classes: int = 4

    # Normalization statistics (from dataset)
    dynamics_mean: Optional[List[float]] = None
    dynamics_std: Optional[List[float]] = None

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        # Auto-compute intermediate feature IDs based on model depth
        if self.intermediate_feature_ids is None:
            # Default to extracting from 3 evenly-spaced layers
            # This will be adjusted in the model based on actual num_blocks
            self.intermediate_feature_ids = [7, 14, 21]


class DynamicsWorldModel(Video2WorldModelRectifiedFlow):
    """Video world model with explicit dynamics prediction.

    This model extends Video2World to jointly predict:
    1. Future video frames using the original rectified flow objective
    2. Per-object dynamics using a DETR-style detection head

    The dynamics prediction enables:
    - Physics-grounded video generation (reduced hallucinations)
    - Counterfactual scenario generation (e.g., "what if car A moved faster")

    Attributes:
        dynamics_head: DETR-style head for object detection and dynamics
        dynamics_embedder: Optional embedder for dynamics conditioning
        dynamics_loss: Combined loss for dynamics prediction
    """

    def __init__(self, config: DynamicsWorldModelConfig):
        """Initialize DynamicsWorldModel.

        Args:
            config: Model configuration
        """
        super().__init__(config)

        self.dynamics_config = config

        # Get model channels from the backbone
        model_channels = self.net.model_channels

        # Create dynamics head configuration
        dynamics_head_config = DynamicsHeadConfig(
            num_object_queries=config.num_object_queries,
            hidden_dim=config.dynamics_hidden_dim,
            num_decoder_layers=config.num_decoder_layers,
            num_heads=config.num_attention_heads,
            dim_feedforward=config.dynamics_feedforward_dim,
            dropout=config.dynamics_dropout,
            num_classes=config.num_classes,
            state_dim=config.state_dim,
        )

        # Create dynamics head
        self.dynamics_head = DynamicsHead(
            config=dynamics_head_config,
            backbone_dim=model_channels,
        )

        # Optional dynamics embedder for conditioning
        self.dynamics_embedder = None
        if config.use_dynamics_conditioning:
            self.dynamics_embedder = DynamicsEmbedder(
                max_agents=config.max_agents,
                state_dim=config.state_dim,
                model_channels=model_channels,
            )

        # Create dynamics loss
        self.dynamics_loss_fn = DynamicsLoss(
            state_weight=1.0,
            class_weight=1.0,
            objectness_weight=1.0,
            kinematic_weight=config.kinematic_loss_weight,
        )

        # Register normalization buffers
        if config.dynamics_mean is not None:
            self.register_buffer(
                'dynamics_mean',
                torch.tensor(config.dynamics_mean, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                'dynamics_mean',
                torch.zeros(config.state_dim, dtype=torch.float32)
            )

        if config.dynamics_std is not None:
            self.register_buffer(
                'dynamics_std',
                torch.tensor(config.dynamics_std, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                'dynamics_std',
                torch.ones(config.state_dim, dtype=torch.float32)
            )

        # Compute intermediate feature IDs based on actual model depth
        self._setup_feature_extraction()

        log.info(
            f"DynamicsWorldModel initialized with "
            f"{config.num_object_queries} object queries, "
            f"extracting features from layers {self._intermediate_feature_ids}"
        )

    def _setup_feature_extraction(self):
        """Setup intermediate feature extraction based on model architecture."""
        # Get number of blocks from the backbone
        num_blocks = len(self.net.blocks)

        # Use config-specified IDs or compute default
        if self.dynamics_config.intermediate_feature_ids is not None:
            self._intermediate_feature_ids = [
                idx for idx in self.dynamics_config.intermediate_feature_ids
                if idx < num_blocks
            ]
        else:
            # Extract from 3 evenly-spaced layers
            self._intermediate_feature_ids = [
                num_blocks // 4,
                num_blocks // 2,
                3 * num_blocks // 4,
            ]

        log.info(f"Using intermediate feature IDs: {self._intermediate_feature_ids}")

    def forward_with_dynamics(
        self,
        data_batch: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Forward pass with both video and dynamics prediction.

        Args:
            data_batch: Batch of data containing:
                - video: Input video tensor
                - agent_dynamics: Ground truth dynamics [B, T, N, state_dim]
                - agent_classes: Ground truth classes [B, T, N]
                - valid_mask: Valid agent mask [B, T, N]
                - frame_dt: Time deltas between frames [B, T-1]

        Returns:
            Tuple of (output_dict, total_loss)
        """
        # Get base video prediction loss
        raw_state, latent_state, condition = self.get_data_and_condition(data_batch)

        # Sample timesteps
        B = latent_state.shape[0]
        timesteps, weights = self.rectified_flow.sample_timesteps(B)

        # Add noise
        noise = torch.randn_like(latent_state)
        xt = self.rectified_flow.add_noise(latent_state, noise, timesteps)

        # Prepare timesteps tensor
        if timesteps.ndim == 1:
            timesteps_B_T = timesteps.unsqueeze(1)
        else:
            timesteps_B_T = timesteps

        # Forward through backbone with intermediate feature extraction
        # Call the network with intermediate_feature_ids to get features
        net_output, intermediate_features = self._forward_backbone_with_features(
            xt, timesteps_B_T, condition
        )

        # Compute video loss (velocity prediction)
        target_velocity = latent_state - noise
        video_loss = torch.mean((net_output - target_velocity) ** 2 * weights.view(-1, 1, 1, 1, 1))

        # Process dynamics if annotations are available
        output_dict = {
            'net_output': net_output,
            'video_loss': video_loss,
        }

        if 'agent_dynamics' in data_batch:
            # Get dynamics predictions from head
            _, _, T, H, W = xt.shape
            # Infer spatial shape from latent
            latent_T = self.config.state_t
            latent_H = H // self.net.patch_spatial
            latent_W = W // self.net.patch_spatial

            dynamics_pred = self.dynamics_head(
                intermediate_features,
                num_frames=latent_T,
                spatial_shape=(latent_H, latent_W),
            )

            # Prepare targets
            targets = {
                'classes': data_batch['agent_classes'],
                'states': data_batch['agent_dynamics'],
                'valid_mask': data_batch['valid_mask'],
            }

            # Compute dynamics loss
            frame_dt = data_batch.get('frame_dt', torch.ones(B, latent_T - 1, device=xt.device) / 10)
            dynamics_losses = self.dynamics_loss_fn(dynamics_pred, targets, frame_dt)

            output_dict.update({
                'dynamics_pred': dynamics_pred,
                **dynamics_losses,
            })

            # Combined loss
            total_loss = (
                self.dynamics_config.video_loss_weight * video_loss +
                self.dynamics_config.dynamics_loss_weight * dynamics_losses['total_dynamics_loss']
            )
        else:
            total_loss = video_loss

        output_dict['total_loss'] = total_loss

        return output_dict, total_loss

    def _forward_backbone_with_features(
        self,
        xt_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        condition,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward through backbone with intermediate feature extraction.

        Args:
            xt_B_C_T_H_W: Noisy input tensor
            timesteps_B_T: Timesteps tensor
            condition: Conditioning information

        Returns:
            Tuple of (net_output, intermediate_features)
        """
        # Call network with intermediate_feature_ids
        result = self.net(
            x_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=timesteps_B_T,
            intermediate_feature_ids=self._intermediate_feature_ids,
            **condition.to_dict(),
        )

        # Handle output format
        if isinstance(result, tuple):
            net_output, intermediate_features = result
        else:
            # No intermediate features returned
            net_output = result
            intermediate_features = []

        return net_output.float(), intermediate_features

    def training_step(
        self,
        data_batch: Dict[str, torch.Tensor],
        iteration: int,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Training step with combined video and dynamics loss.

        Args:
            data_batch: Batch of training data
            iteration: Current training iteration

        Returns:
            Tuple of (output_dict, loss)
        """
        output_dict, loss = self.forward_with_dynamics(data_batch)

        # Log metrics
        self._log_training_metrics(output_dict, iteration)

        return output_dict, loss

    def _log_training_metrics(
        self,
        output_dict: Dict[str, torch.Tensor],
        iteration: int,
    ):
        """Log training metrics."""
        if iteration % 100 == 0:
            metrics = {}
            for key in ['video_loss', 'total_dynamics_loss', 'state_loss',
                       'class_loss', 'kinematic_loss']:
                if key in output_dict:
                    metrics[key] = output_dict[key].item()

            if metrics:
                log.info(f"Iteration {iteration}: {metrics}")

    def predict_dynamics(
        self,
        video_latents: torch.Tensor,
        timesteps: torch.Tensor,
        condition,
    ) -> Dict[str, torch.Tensor]:
        """Predict dynamics for given video latents.

        Args:
            video_latents: Video latent tensor [B, C, T, H, W]
            timesteps: Timesteps tensor
            condition: Conditioning information

        Returns:
            Dictionary with dynamics predictions
        """
        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(1)

        # Forward through backbone
        _, intermediate_features = self._forward_backbone_with_features(
            video_latents, timesteps, condition
        )

        # Predict dynamics
        _, _, T, H, W = video_latents.shape
        latent_T = self.config.state_t
        latent_H = H // self.net.patch_spatial
        latent_W = W // self.net.patch_spatial

        dynamics_pred = self.dynamics_head(
            intermediate_features,
            num_frames=latent_T,
            spatial_shape=(latent_H, latent_W),
        )

        # Denormalize predictions
        dynamics_pred['state_pred_denorm'] = self._denormalize_dynamics(
            dynamics_pred['state_pred']
        )

        return dynamics_pred

    def _denormalize_dynamics(self, normalized_states: torch.Tensor) -> torch.Tensor:
        """Denormalize dynamics predictions.

        Args:
            normalized_states: Normalized state predictions [B, T, N, state_dim]

        Returns:
            Denormalized states in original units
        """
        return normalized_states * self.dynamics_std + self.dynamics_mean

    def generate_with_dynamics(
        self,
        data_batch: Dict[str, torch.Tensor],
        num_steps: int = 35,
        guidance: float = 1.5,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Generate video with dynamics predictions.

        Args:
            data_batch: Input data batch
            num_steps: Number of sampling steps
            guidance: Classifier-free guidance scale

        Returns:
            Tuple of (generated_video, dynamics_predictions)
        """
        # Get velocity function with guidance
        velocity_fn = self.get_velocity_fn_from_batch(data_batch, guidance=guidance)

        # Get initial latent shape
        _, x0, condition = self.get_data_and_condition(data_batch)
        B, C, T, H, W = x0.shape

        # Initialize from noise
        xt = torch.randn_like(x0)

        # Setup scheduler
        self.sample_scheduler.set_timesteps(num_steps, device=x0.device)
        timesteps = self.sample_scheduler.timesteps

        # Sampling loop with dynamics prediction at each step
        all_dynamics = []

        for i, t in enumerate(timesteps):
            timestep = t.expand(B)
            noise = torch.randn_like(xt)  # For velocity computation

            # Predict velocity
            velocity = velocity_fn(noise, xt, timestep)

            # Optionally predict dynamics at this step
            if i % (len(timesteps) // 5 + 1) == 0:  # Sample dynamics every few steps
                with torch.no_grad():
                    _, condition_clean, _ = self.get_data_and_condition(data_batch)
                    dynamics = self.predict_dynamics(xt, timestep, condition_clean)
                    all_dynamics.append(dynamics)

            # Update sample
            xt = self.sample_scheduler.step(velocity, t, xt).prev_sample

        # Final dynamics prediction
        final_timestep = torch.zeros(B, device=x0.device)
        final_dynamics = self.predict_dynamics(xt, final_timestep, condition)

        # Decode to video
        generated_video = self.tokenizer.decode(xt)

        return generated_video, final_dynamics
