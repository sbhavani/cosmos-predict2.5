# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Video dataset with H5 kinematics data for dynamics-aware world models."""

import os
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import default_collate

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import VideoDataset


class VideoKinematicsDataset(VideoDataset):
    """Extends VideoDataset to load H5 kinematics alongside videos.

    The kinematics H5 files should have the following structure:
    - frames: Shape (T, max_agents, 13) containing per-frame, per-agent:
        - position (x, y, z) in meters relative to camera
        - velocity (vx, vy, vz)
        - acceleration (ax, ay, az)
        - class one-hot (4 classes)
    - num_agents: Shape (T,) - number of valid agents per frame
    - timestamps: Shape (T,) - timestamp for each frame in microseconds

    Attributes:
        kinematics_subdir: Subdirectory containing H5 kinematics files
        max_agents: Maximum number of agents per frame
        state_dim: Dimension of state vector (9: position + velocity + acceleration)
        num_classes: Number of object classes
        dynamics_mean: Mean for state normalization
        dynamics_std: Std for state normalization
    """

    def __init__(
        self,
        dataset_dir: str,
        num_frames: int,
        video_size: tuple[int, int],
        kinematics_subdir: str = "kinematics",
        max_agents: int = 32,
        state_dim: int = 9,
        num_classes: int = 4,
        dynamics_mean: Optional[list[float]] = None,
        dynamics_std: Optional[list[float]] = None,
        normalize_dynamics: bool = True,
        **kwargs,
    ) -> None:
        """Initialize VideoKinematicsDataset.

        Args:
            dataset_dir: Base path to the dataset directory
            num_frames: Number of frames to load per sequence
            video_size: Target size (H, W) for video frames
            kinematics_subdir: Subdirectory name for kinematics H5 files
            max_agents: Maximum number of agents to track
            state_dim: Dimension of dynamics state (default 9: xyz + velocity + accel)
            num_classes: Number of object classes
            dynamics_mean: Mean values for normalizing dynamics (length state_dim)
            dynamics_std: Std values for normalizing dynamics (length state_dim)
            normalize_dynamics: Whether to apply normalization to dynamics
            **kwargs: Additional arguments passed to VideoDataset
        """
        super().__init__(
            dataset_dir=dataset_dir,
            num_frames=num_frames,
            video_size=video_size,
            **kwargs,
        )

        self.kinematics_dir = os.path.join(dataset_dir, kinematics_subdir)
        self.max_agents = max_agents
        self.state_dim = state_dim
        self.num_classes = num_classes
        self.normalize_dynamics = normalize_dynamics

        # Normalization statistics
        if dynamics_mean is not None:
            self.dynamics_mean = np.array(dynamics_mean, dtype=np.float32)
        else:
            self.dynamics_mean = np.zeros(state_dim, dtype=np.float32)

        if dynamics_std is not None:
            self.dynamics_std = np.array(dynamics_std, dtype=np.float32)
        else:
            self.dynamics_std = np.ones(state_dim, dtype=np.float32)

        # Validate kinematics directory exists
        if not os.path.exists(self.kinematics_dir):
            log.warning(
                f"Kinematics directory {self.kinematics_dir} does not exist. "
                "Dataset will return empty kinematics."
            )

        # Track frame indices for alignment (set during _load_video)
        self._last_start_frame = 0
        self._last_end_frame = 0

    def _load_video(self, video_path: str) -> tuple[np.ndarray, float]:
        """Load video and track frame indices for kinematics alignment."""
        from decord import VideoReader, cpu

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        total_frames = len(vr)

        if total_frames < self.sequence_length:
            raise ValueError(
                f"Video {video_path} has only {total_frames} frames, "
                f"at least {self.sequence_length} frames are required."
            )

        # Randomly sample a sequence of frames
        max_start_idx = total_frames - self.sequence_length
        start_frame = np.random.randint(0, max_start_idx + 1)
        end_frame = start_frame + self.sequence_length
        frame_ids = np.arange(start_frame, end_frame).tolist()

        # Store for kinematics alignment
        self._last_start_frame = start_frame
        self._last_end_frame = end_frame

        frame_data = vr.get_batch(frame_ids).asnumpy()
        vr.seek(0)

        try:
            fps = vr.get_avg_fps()
        except Exception:
            fps = 16.0

        del vr
        return frame_data, fps

    def _load_kinematics(self, video_path: str, start_frame: int, num_frames: int) -> dict[str, torch.Tensor]:
        """Load H5 kinematics for corresponding video segment.

        Args:
            video_path: Path to the video file (used to find corresponding H5)
            start_frame: Starting frame index
            num_frames: Number of frames to load

        Returns:
            Dictionary with kinematics tensors:
                - agent_dynamics: [T, max_agents, state_dim] dynamics states
                - agent_classes: [T, max_agents] class indices
                - num_agents: [T] number of valid agents per frame
                - frame_dt: [T-1] time deltas between consecutive frames
                - timestamps: [T] timestamps for each frame
                - valid_mask: [T, max_agents] boolean mask for valid agents
        """
        video_name = os.path.basename(video_path).replace('.mp4', '')
        h5_path = os.path.join(self.kinematics_dir, f"{video_name}.h5")

        # Default empty tensors if H5 file doesn't exist
        if not os.path.exists(h5_path):
            return self._get_empty_kinematics(num_frames)

        try:
            with h5py.File(h5_path, 'r') as f:
                # Get total frames in H5 and validate range
                h5_total_frames = f['frames'].shape[0]

                # Clamp frame range to H5 bounds
                actual_start = min(start_frame, h5_total_frames - num_frames)
                actual_start = max(0, actual_start)
                actual_end = min(actual_start + num_frames, h5_total_frames)

                # Load data slices
                # frames: [T_total, max_agents, 13] -> position(3) + velocity(3) + accel(3) + class(4)
                all_frames = f['frames'][actual_start:actual_end]
                num_agents_per_frame = f['num_agents'][actual_start:actual_end]
                timestamps = f['timestamps'][actual_start:actual_end]

                # Handle case where H5 has fewer frames than requested
                actual_loaded = actual_end - actual_start
                if actual_loaded < num_frames:
                    # Pad with last frame
                    pad_count = num_frames - actual_loaded
                    all_frames = np.concatenate([
                        all_frames,
                        np.tile(all_frames[-1:], (pad_count, 1, 1))
                    ], axis=0)
                    num_agents_per_frame = np.concatenate([
                        num_agents_per_frame,
                        np.tile(num_agents_per_frame[-1:], pad_count)
                    ])
                    timestamps = np.concatenate([
                        timestamps,
                        timestamps[-1:] + np.arange(1, pad_count + 1) * 1000000  # Assume 1s intervals
                    ])

            # Extract state (9 dims: position, velocity, acceleration)
            agent_states = all_frames[:, :, :self.state_dim]  # [T, max_agents, 9]

            # Extract classes from one-hot encoding
            class_onehot = all_frames[:, :, self.state_dim:self.state_dim + self.num_classes]
            agent_classes = class_onehot.argmax(axis=-1)  # [T, max_agents]

            # Create valid mask based on num_agents
            T, N = agent_states.shape[:2]
            valid_mask = np.zeros((T, N), dtype=np.float32)
            for t in range(T):
                valid_mask[t, :num_agents_per_frame[t]] = 1.0

            # Normalize dynamics
            if self.normalize_dynamics:
                agent_states = (agent_states - self.dynamics_mean) / (self.dynamics_std + 1e-8)

            # Compute dt between frames (in seconds, assuming timestamps in microseconds)
            frame_dt = np.diff(timestamps).astype(np.float32) / 1e6  # Convert to seconds

            return {
                'agent_dynamics': torch.from_numpy(agent_states.astype(np.float32)),
                'agent_classes': torch.from_numpy(agent_classes.astype(np.int64)),
                'num_agents': torch.from_numpy(num_agents_per_frame.astype(np.int32)),
                'frame_dt': torch.from_numpy(frame_dt),
                'timestamps': torch.from_numpy(timestamps.astype(np.float32)),
                'valid_mask': torch.from_numpy(valid_mask),
            }

        except Exception as e:
            log.warning(f"Failed to load kinematics from {h5_path}: {e}")
            return self._get_empty_kinematics(num_frames)

    def _get_empty_kinematics(self, num_frames: int) -> dict[str, torch.Tensor]:
        """Return empty kinematics tensors when H5 file is not available."""
        return {
            'agent_dynamics': torch.zeros(num_frames, self.max_agents, self.state_dim),
            'agent_classes': torch.zeros(num_frames, self.max_agents, dtype=torch.long),
            'num_agents': torch.zeros(num_frames, dtype=torch.int32),
            'frame_dt': torch.ones(num_frames - 1) / 10.0,  # Assume 10 FPS
            'timestamps': torch.arange(num_frames, dtype=torch.float32),
            'valid_mask': torch.zeros(num_frames, self.max_agents),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get video and kinematics data for a sample.

        Returns:
            Dictionary containing:
                - video: RGB frames tensor [C, T, H, W]
                - ai_caption: Text caption
                - fps: Video frame rate
                - agent_dynamics: [T, max_agents, state_dim] dynamics states
                - agent_classes: [T, max_agents] class indices
                - num_agents: [T] number of valid agents per frame
                - frame_dt: [T-1] time deltas between consecutive frames
                - valid_mask: [T, max_agents] boolean mask for valid agents
        """
        # Get video data from parent class
        data = super().__getitem__(index)

        # Load corresponding kinematics using tracked frame indices
        video_path = self.video_paths[index]
        kinematics = self._load_kinematics(
            video_path,
            self._last_start_frame,
            self.sequence_length
        )
        data.update(kinematics)

        return data


def kinematics_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collate function that handles variable agent counts.

    Pads agent tensors to the maximum number of agents in the batch.

    Args:
        batch: List of sample dictionaries

    Returns:
        Collated batch dictionary
    """
    # Separate video data from kinematics
    video_keys = ['video', 'ai_caption', 'fps', 'image_size', 'num_frames', 'padding_mask']
    kinematics_keys = ['agent_dynamics', 'agent_classes', 'num_agents', 'frame_dt', 'timestamps', 'valid_mask']

    # Standard collation for video data
    video_batch = {k: [item[k] for item in batch] for k in video_keys if k in batch[0]}

    # Handle each video key appropriately
    collated = {}
    for k, v in video_batch.items():
        if k == 'ai_caption':
            collated[k] = v  # Keep as list of strings
        else:
            collated[k] = default_collate(v)

    # Collate kinematics with proper padding
    B = len(batch)
    T = batch[0]['agent_dynamics'].shape[0]
    max_agents = max(item['num_agents'].max().item() for item in batch if item['num_agents'].sum() > 0)
    max_agents = max(max_agents, 1)  # At least 1

    # Determine actual max_agents from data
    data_max_agents = batch[0]['agent_dynamics'].shape[1]

    # Initialize padded tensors
    state_dim = batch[0]['agent_dynamics'].shape[-1]
    agent_dynamics = torch.zeros(B, T, data_max_agents, state_dim)
    agent_classes = torch.zeros(B, T, data_max_agents, dtype=torch.long) - 1  # -1 = no object
    num_agents = torch.stack([item['num_agents'] for item in batch])
    valid_mask = torch.zeros(B, T, data_max_agents)
    frame_dt = torch.stack([item['frame_dt'] for item in batch])
    timestamps = torch.stack([item['timestamps'] for item in batch])

    for b, item in enumerate(batch):
        n = data_max_agents
        agent_dynamics[b] = item['agent_dynamics']
        agent_classes[b] = item['agent_classes']
        valid_mask[b] = item['valid_mask']

    collated['agent_dynamics'] = agent_dynamics
    collated['agent_classes'] = agent_classes
    collated['num_agents'] = num_agents
    collated['valid_mask'] = valid_mask
    collated['frame_dt'] = frame_dt
    collated['timestamps'] = timestamps

    return collated
