#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compute normalization statistics for dynamics data from H5 files.

This script computes mean and standard deviation for the dynamics state
(position, velocity, acceleration) from all H5 kinematics files in a dataset.

Usage:
    python scripts/compute_dynamics_stats.py --dataset_dir datasetWM

    # Or with uv:
    uv run scripts/compute_dynamics_stats.py --dataset_dir datasetWM
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np


def load_all_dynamics(kinematics_dir: str, state_dim: int = 9) -> Tuple[np.ndarray, np.ndarray]:
    """Load all dynamics data from H5 files.

    Args:
        kinematics_dir: Directory containing H5 kinematics files
        state_dim: Number of state dimensions (default 9: pos + vel + accel)

    Returns:
        Tuple of (all_states, valid_mask) where:
            - all_states: [total_samples, state_dim] flattened valid states
            - valid_mask: Number of valid samples
    """
    all_states = []

    h5_files = sorted(Path(kinematics_dir).glob("*.h5"))
    print(f"Found {len(h5_files)} H5 files")

    for h5_path in h5_files:
        print(f"  Processing {h5_path.name}...")

        with h5py.File(h5_path, 'r') as f:
            frames = f['frames'][:]  # [T, max_agents, 13]
            num_agents = f['num_agents'][:]  # [T]

            T, N, D = frames.shape

            # Extract state dimensions (first 9: position + velocity + acceleration)
            states = frames[:, :, :state_dim]  # [T, N, 9]

            # Only include valid agents
            for t in range(T):
                n_valid = num_agents[t]
                if n_valid > 0:
                    valid_states = states[t, :n_valid]  # [n_valid, 9]
                    all_states.append(valid_states)

    if len(all_states) == 0:
        print("Warning: No valid states found!")
        return np.zeros((1, state_dim)), 0

    all_states = np.concatenate(all_states, axis=0)  # [total_valid, 9]
    print(f"Total valid agent-frames: {len(all_states)}")

    return all_states, len(all_states)


def compute_statistics(states: np.ndarray) -> Dict[str, List[float]]:
    """Compute mean and std for each state dimension.

    Args:
        states: [N, state_dim] array of states

    Returns:
        Dictionary with 'mean' and 'std' lists
    """
    mean = np.mean(states, axis=0)
    std = np.std(states, axis=0)

    # Avoid division by zero - set minimum std
    std = np.maximum(std, 1e-6)

    return {
        'mean': mean.tolist(),
        'std': std.tolist(),
    }


def compute_per_dimension_stats(states: np.ndarray) -> Dict[str, Dict]:
    """Compute detailed statistics for each dimension.

    Args:
        states: [N, state_dim] array of states

    Returns:
        Dictionary with per-dimension statistics
    """
    dim_names = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az']

    stats = {}
    for i, name in enumerate(dim_names):
        if i < states.shape[1]:
            dim_data = states[:, i]
            stats[name] = {
                'mean': float(np.mean(dim_data)),
                'std': float(np.std(dim_data)),
                'min': float(np.min(dim_data)),
                'max': float(np.max(dim_data)),
                'median': float(np.median(dim_data)),
            }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Compute dynamics normalization statistics")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to dataset directory (containing kinematics/ subdirectory)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file (default: dataset_dir/dynamics_stats.json)")
    parser.add_argument("--state_dim", type=int, default=9,
                        help="Number of state dimensions (default: 9)")
    args = parser.parse_args()

    kinematics_dir = os.path.join(args.dataset_dir, "kinematics")

    if not os.path.exists(kinematics_dir):
        print(f"Error: Kinematics directory not found: {kinematics_dir}")
        return 1

    print(f"Computing statistics from: {kinematics_dir}")
    print(f"State dimensions: {args.state_dim}")
    print()

    # Load all dynamics data
    states, num_samples = load_all_dynamics(kinematics_dir, args.state_dim)

    if num_samples == 0:
        print("Error: No valid samples found!")
        return 1

    print()

    # Compute statistics
    stats = compute_statistics(states)
    per_dim_stats = compute_per_dimension_stats(states)

    # Print results
    print("=" * 60)
    print("NORMALIZATION STATISTICS")
    print("=" * 60)
    print()

    dim_names = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az']

    print("Mean:")
    for i, (name, val) in enumerate(zip(dim_names, stats['mean'])):
        print(f"  {name:3s}: {val:12.6f}")

    print()
    print("Std:")
    for i, (name, val) in enumerate(zip(dim_names, stats['std'])):
        print(f"  {name:3s}: {val:12.6f}")

    print()
    print("Per-dimension statistics:")
    for name, dim_stats in per_dim_stats.items():
        print(f"  {name:3s}: min={dim_stats['min']:10.4f}, max={dim_stats['max']:10.4f}, "
              f"mean={dim_stats['mean']:10.4f}, std={dim_stats['std']:10.4f}")

    print()
    print("=" * 60)
    print("PYTHON CONFIG VALUES")
    print("=" * 60)
    print()
    print("# Add these to your DynamicsWorldModelConfig:")
    print(f"dynamics_mean = {stats['mean']}")
    print(f"dynamics_std = {stats['std']}")

    # Save to JSON
    output_path = args.output or os.path.join(args.dataset_dir, "dynamics_stats.json")

    output_data = {
        'num_samples': num_samples,
        'state_dim': args.state_dim,
        'mean': stats['mean'],
        'std': stats['std'],
        'per_dimension': per_dim_stats,
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print()
    print(f"Statistics saved to: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
