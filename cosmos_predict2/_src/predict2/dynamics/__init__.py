# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Dynamics-aware world model extension for Cosmos Predict 2.5.

This module adds explicit ego-vehicle and agent dynamics prediction to reduce
hallucinations and enable counterfactual scenario editing.

Example usage:

    # Training
    from cosmos_predict2._src.predict2.dynamics import DynamicsWorldModel, DynamicsWorldModelConfig
    from cosmos_predict2._src.predict2.dynamics.datasets import VideoKinematicsDataset

    config = DynamicsWorldModelConfig(
        num_object_queries=32,
        dynamics_loss_weight=0.1,
        kinematic_loss_weight=0.01,
    )
    model = DynamicsWorldModel(config)

    dataset = VideoKinematicsDataset(
        dataset_dir="path/to/dataset",
        num_frames=32,
        video_size=(704, 1280),
    )

    # Inference with counterfactual editing
    from cosmos_predict2._src.predict2.dynamics.inference import DynamicsInference

    inference = DynamicsInference(model)
    result = inference.generate_counterfactual(
        video_frames,
        agent_modifications={
            0: {'velocity_delta': (2.0, 0.0, 0.0)}  # Car A moves 2 m/s faster
        }
    )
"""

# Modules
from cosmos_predict2._src.predict2.dynamics.modules.dynamics_head import (
    DynamicsHead,
    DynamicsHeadConfig,
)
from cosmos_predict2._src.predict2.dynamics.modules.dynamics_embedder import (
    DynamicsEmbedder,
    TemporalDynamicsEmbedder,
    CounterfactualDynamicsEmbedder,
)

# Losses
from cosmos_predict2._src.predict2.dynamics.losses.dynamics_loss import DynamicsLoss
from cosmos_predict2._src.predict2.dynamics.losses.kinematic_loss import (
    KinematicConstraintLoss,
    SmoothMotionLoss,
    CollisionAvoidanceLoss,
)
from cosmos_predict2._src.predict2.dynamics.losses.hungarian_matcher import (
    HungarianMatcher,
    HungarianMatcherBatched,
)

# Models
from cosmos_predict2._src.predict2.dynamics.models.dynamics_world_model import (
    DynamicsWorldModel,
    DynamicsWorldModelConfig,
)

# Datasets
from cosmos_predict2._src.predict2.dynamics.datasets.h5_kinematics_dataset import (
    VideoKinematicsDataset,
    kinematics_collate_fn,
)

# Conditioning
from cosmos_predict2._src.predict2.dynamics.conditioner.dynamics_condition import (
    DynamicsCondition,
    create_dynamics_condition_from_video2world,
)

# Inference
from cosmos_predict2._src.predict2.dynamics.inference.dynamics_inference import (
    DynamicsInference,
    load_dynamics_model,
)

__all__ = [
    # Modules
    "DynamicsHead",
    "DynamicsHeadConfig",
    "DynamicsEmbedder",
    "TemporalDynamicsEmbedder",
    "CounterfactualDynamicsEmbedder",
    # Losses
    "DynamicsLoss",
    "KinematicConstraintLoss",
    "SmoothMotionLoss",
    "CollisionAvoidanceLoss",
    "HungarianMatcher",
    "HungarianMatcherBatched",
    # Models
    "DynamicsWorldModel",
    "DynamicsWorldModelConfig",
    # Datasets
    "VideoKinematicsDataset",
    "kinematics_collate_fn",
    # Conditioning
    "DynamicsCondition",
    "create_dynamics_condition_from_video2world",
    # Inference
    "DynamicsInference",
    "load_dynamics_model",
]
