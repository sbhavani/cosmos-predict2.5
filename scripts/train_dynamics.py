#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Training script for dynamics-aware world model.

This script demonstrates how to fine-tune a DynamicsWorldModel on a custom
dataset with kinematics annotations.

Usage:
    # Basic training with pretrained checkpoint
    python scripts/train_dynamics.py \
        --dataset_dir datasetWM \
        --output_dir outputs/dynamics_training \
        --pretrained_checkpoint checkpoints/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt \
        --tokenizer_checkpoint checkpoints/tokenizer.pth

    # Dry run test
    python scripts/train_dynamics.py \
        --dataset_dir datasetWM \
        --output_dir outputs/dynamics_training \
        --pretrained_checkpoint checkpoints/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt \
        --tokenizer_checkpoint checkpoints/tokenizer.pth \
        --dry_run

    # Full training with custom settings
    python scripts/train_dynamics.py \
        --dataset_dir datasetWM \
        --output_dir outputs/dynamics_training \
        --pretrained_checkpoint checkpoints/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt \
        --tokenizer_checkpoint checkpoints/tokenizer.pth \
        --batch_size 1 \
        --num_frames 32 \
        --max_iterations 10000 \
        --learning_rate 1e-5 \
        --dynamics_loss_weight 0.1

Note: This script requires a GPU with sufficient memory (recommended: 24GB+).
For full training, use the Hydra-based configs in the configs/ directory.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(description="Train dynamics-aware world model")

    # Data
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="outputs/dynamics_training",
                        help="Output directory for checkpoints and logs")

    # Model
    parser.add_argument("--num_object_queries", type=int, default=32,
                        help="Number of object queries for DETR-style detection")
    parser.add_argument("--dynamics_hidden_dim", type=int, default=256,
                        help="Hidden dimension for dynamics head")
    parser.add_argument("--num_decoder_layers", type=int, default=6,
                        help="Number of transformer decoder layers")

    # Loss weights
    parser.add_argument("--video_loss_weight", type=float, default=1.0,
                        help="Weight for video prediction loss")
    parser.add_argument("--dynamics_loss_weight", type=float, default=0.1,
                        help="Weight for dynamics prediction loss")
    parser.add_argument("--kinematic_loss_weight", type=float, default=0.01,
                        help="Weight for kinematic constraint loss")

    # Training
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size per GPU")
    parser.add_argument("--num_frames", type=int, default=32,
                        help="Number of frames per video clip")
    parser.add_argument("--video_height", type=int, default=704,
                        help="Video height")
    parser.add_argument("--video_width", type=int, default=1280,
                        help="Video width")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--max_iterations", type=int, default=10000,
                        help="Maximum training iterations")
    parser.add_argument("--save_interval", type=int, default=1000,
                        help="Save checkpoint every N iterations")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log metrics every N iterations")

    # Optimizer
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Number of warmup steps")

    # Checkpoint
    parser.add_argument("--pretrained_checkpoint", type=str, default=None,
                        help="Path to pretrained Cosmos 2.5 checkpoint")
    parser.add_argument("--tokenizer_checkpoint", type=str, default=None,
                        help="Path to tokenizer checkpoint (tokenizer.pth)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume training from checkpoint")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")

    # Debugging
    parser.add_argument("--dry_run", action="store_true",
                        help="Run one iteration and exit (for testing)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with extra logging")

    return parser.parse_args()


def load_dynamics_stats(dataset_dir: str) -> Dict[str, list]:
    """Load normalization statistics from dataset."""
    stats_path = os.path.join(dataset_dir, "dynamics_stats.json")

    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        return {
            'mean': stats['mean'],
            'std': stats['std'],
        }

    # Default to identity normalization if stats don't exist
    print(f"Warning: {stats_path} not found. Using identity normalization.")
    return {
        'mean': [0.0] * 9,
        'std': [1.0] * 9,
    }


def create_dataloader(args):
    """Create training dataloader."""
    from cosmos_predict2._src.predict2.dynamics.datasets.h5_kinematics_dataset import (
        VideoKinematicsDataset,
        kinematics_collate_fn,
    )

    # Load normalization stats
    stats = load_dynamics_stats(args.dataset_dir)

    dataset = VideoKinematicsDataset(
        dataset_dir=args.dataset_dir,
        num_frames=args.num_frames,
        video_size=(args.video_height, args.video_width),
        dynamics_mean=stats['mean'],
        dynamics_std=stats['std'],
        normalize_dynamics=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=kinematics_collate_fn,
    )

    return dataloader


def create_model(args):
    """Create dynamics world model.

    This function creates a DynamicsWorldModel with proper configuration including:
    - Tokenizer: Wan2pt1 VAE for video encoding/decoding
    - Net: MinimalV1LVGDiT backbone (2B configuration)
    - Conditioner: Video prediction conditioner
    - Dynamics head: DETR-style detection head for physics prediction
    """
    from cosmos_predict2._src.predict2.dynamics.models.dynamics_world_model import (
        DynamicsWorldModel,
        DynamicsWorldModelConfig,
    )
    from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
    from cosmos_predict2._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface
    from cosmos_predict2._src.predict2.networks.minimal_v1_lvg_dit import MinimalV1LVGDiT
    from cosmos_predict2._src.predict2.networks.minimal_v4_dit import SACConfig
    from cosmos_predict2._src.predict2.conditioner import (
        ReMapkey,
        TextAttr,
        BooleanFlag,
    )
    from cosmos_predict2._src.predict2.configs.video2world.defaults.conditioner import (
        Video2WorldConditioner,
    )

    # Load dynamics stats
    stats = load_dynamics_stats(args.dataset_dir)

    # Determine tokenizer checkpoint path
    tokenizer_path = args.tokenizer_checkpoint
    if tokenizer_path is None and args.pretrained_checkpoint:
        # Try to find tokenizer.pth in the same directory as pretrained checkpoint
        checkpoint_dir = os.path.dirname(args.pretrained_checkpoint)
        parent_dir = os.path.dirname(checkpoint_dir)
        potential_path = os.path.join(parent_dir, "tokenizer.pth")
        if os.path.exists(potential_path):
            tokenizer_path = potential_path
            print(f"  Found tokenizer at: {tokenizer_path}")

    if tokenizer_path is None:
        raise ValueError(
            "Tokenizer checkpoint is required. Please provide --tokenizer_checkpoint "
            "or ensure tokenizer.pth exists alongside the pretrained checkpoint."
        )

    # Create tokenizer config with local checkpoint path
    tokenizer_config = L(Wan2pt1VAEInterface)(
        name="wan2pt1_tokenizer",
        vae_pth=tokenizer_path,
        chunk_duration=81,
    )

    # Create net config for Cosmos-Predict2.5-2B model
    # This matches the architecture of the pretrained checkpoint
    net_config = L(MinimalV1LVGDiT)(
        max_img_h=240,
        max_img_w=240,
        max_frames=128,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        model_channels=2048,  # 2B model
        num_blocks=28,
        num_heads=16,  # 2B model
        concat_padding_mask=True,
        pos_emb_cls="rope3d",
        pos_emb_learnable=True,
        pos_emb_interpolation="crop",
        use_adaln_lora=True,
        adaln_lora_dim=256,
        atten_backend="minimal_a2a",
        extra_per_block_abs_pos_emb=False,  # 2B model
        rope_h_extrapolation_ratio=1.0,
        rope_w_extrapolation_ratio=1.0,
        rope_t_extrapolation_ratio=1.0,  # 2B model
        sac_config=SACConfig(),
    )

    # Create a minimal conditioner config
    # This handles text embeddings and other conditioning signals
    conditioner_config = L(Video2WorldConditioner)(
        fps=L(ReMapkey)(
            input_key="fps",
            output_key="fps",
            dropout_rate=0.0,
            dtype=None,
        ),
        padding_mask=L(ReMapkey)(
            input_key="padding_mask",
            output_key="padding_mask",
            dropout_rate=0.0,
            dtype=None,
        ),
        text=L(TextAttr)(
            input_key=["t5_text_embeddings"],
            dropout_rate=0.2,
            use_empty_string=False,
        ),
        use_video_condition=L(BooleanFlag)(
            input_key="fps",
            output_key="use_video_condition",
            dropout_rate=0.2,
        ),
    )

    config = DynamicsWorldModelConfig(
        # Model components
        tokenizer=tokenizer_config,
        net=net_config,
        conditioner=conditioner_config,

        # Dynamics head settings
        num_object_queries=args.num_object_queries,
        dynamics_hidden_dim=args.dynamics_hidden_dim,
        num_decoder_layers=args.num_decoder_layers,

        # Loss weights
        video_loss_weight=args.video_loss_weight,
        dynamics_loss_weight=args.dynamics_loss_weight,
        kinematic_loss_weight=args.kinematic_loss_weight,

        # Normalization
        dynamics_mean=stats['mean'],
        dynamics_std=stats['std'],
    )

    model = DynamicsWorldModel(config)

    # Load pretrained checkpoint if specified
    if args.pretrained_checkpoint:
        print(f"Loading pretrained checkpoint: {args.pretrained_checkpoint}")
        checkpoint = torch.load(args.pretrained_checkpoint, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Load with strict=False to allow new dynamics head parameters
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  Missing keys: {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")

    return model


def create_optimizer(model, args):
    """Create optimizer with separate learning rates for backbone and dynamics head."""
    # Separate parameters
    backbone_params = []
    dynamics_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'dynamics_head' in name or 'dynamics_embedder' in name:
                dynamics_params.append(param)
            else:
                backbone_params.append(param)

    # Higher learning rate for new dynamics parameters
    param_groups = [
        {'params': backbone_params, 'lr': args.learning_rate},
        {'params': dynamics_params, 'lr': args.learning_rate * 10},
    ]

    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)

    return optimizer


def train_step(model, batch, optimizer, device, iteration, args):
    """Run one training step."""
    # Move batch to device
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    # Forward pass
    output_dict, loss = model.forward_with_dynamics(batch)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    return output_dict, loss.item()


def save_checkpoint(model, optimizer, iteration, output_dir, prefix="checkpoint"):
    """Save training checkpoint."""
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, f"{prefix}_iter_{iteration:06d}.pt")

    torch.save({
        'iteration': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, checkpoint_path)

    print(f"Saved checkpoint: {checkpoint_path}")


def main():
    args = parse_args()

    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 60)
    print("Dynamics-Aware World Model Training")
    print("=" * 60)
    print(f"Dataset: {args.dataset_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print()

    # Check device
    device = torch.device(args.device)
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Create dataloader
    print("Creating dataloader...")
    try:
        dataloader = create_dataloader(args)
        print(f"  Dataset size: {len(dataloader.dataset)} videos")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Iterations per epoch: {len(dataloader)}")
    except Exception as e:
        print(f"Error creating dataloader: {e}")
        print("\nThis script requires the full cosmos_predict2 environment.")
        print("For a minimal test, use the test_dynamics_components.py script instead.")
        return 1

    # Create model
    print("\nCreating model...")
    try:
        model = create_model(args)
        model = model.to(device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params / 1e6:.1f}M")
        print(f"  Trainable parameters: {trainable_params / 1e6:.1f}M")
    except Exception as e:
        print(f"Error creating model: {e}")
        print("\nThe full model requires pretrained Cosmos 2.5 components.")
        return 1

    # Create optimizer
    print("\nCreating optimizer...")
    optimizer = create_optimizer(model, args)
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay: {args.weight_decay}")

    # Resume from checkpoint if specified
    start_iteration = 0
    if args.resume_from:
        print(f"\nResuming from: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_iteration = checkpoint['iteration']
        print(f"  Resumed at iteration {start_iteration}")

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    model.train()
    data_iter = iter(dataloader)
    start_time = datetime.now()

    for iteration in range(start_iteration, args.max_iterations):
        # Get next batch (cycle through dataset)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Training step
        try:
            output_dict, loss = train_step(model, batch, optimizer, device, iteration, args)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\nOOM at iteration {iteration}. Try reducing batch_size or num_frames.")
                torch.cuda.empty_cache()
                continue
            raise

        # Logging
        if iteration % args.log_interval == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            iter_per_sec = (iteration - start_iteration + 1) / elapsed if elapsed > 0 else 0

            log_str = f"Iter {iteration:6d} | Loss: {loss:.4f}"

            if 'video_loss' in output_dict:
                log_str += f" | Video: {output_dict['video_loss'].item():.4f}"

            if 'total_dynamics_loss' in output_dict:
                log_str += f" | Dynamics: {output_dict['total_dynamics_loss'].item():.4f}"

            if 'kinematic_loss' in output_dict:
                log_str += f" | Kinematic: {output_dict['kinematic_loss'].item():.4f}"

            log_str += f" | {iter_per_sec:.2f} it/s"

            print(log_str)

        # Save checkpoint
        if iteration > 0 and iteration % args.save_interval == 0:
            save_checkpoint(model, optimizer, iteration, args.output_dir)

        # Dry run exit
        if args.dry_run:
            print("\nDry run complete!")
            save_checkpoint(model, optimizer, iteration, args.output_dir, prefix="dry_run")
            return 0

    # Final checkpoint
    save_checkpoint(model, optimizer, args.max_iterations, args.output_dir, prefix="final")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
