"""Utilities for loading and preprocessing AIST++ 2D keypoints."""

import pickle

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema


NUM_JOINTS = 17
NUM_KEYPOINT_CHANNELS = 3


def _validate_keypoints(keypoints, *, name="keypoints"):
    array = np.asarray(keypoints)
    if array.ndim != 3 or array.shape[1:] != (
        NUM_JOINTS,
        NUM_KEYPOINT_CHANNELS,
    ):
        raise ValueError(
            f"{name} must have shape [frames, {NUM_JOINTS}, "
            f"{NUM_KEYPOINT_CHANNELS}], got {array.shape}"
        )
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one frame")
    return array


def load_keypoints(keypoints_path):
    with open(keypoints_path, "rb") as file:
        keypoints = pickle.load(file)
    if not isinstance(keypoints, np.ndarray):
        raise TypeError("keypoints must be stored as a numpy array")
    return _validate_keypoints(keypoints)


def interp_nan_keypoints(kps):
    keypoints = _validate_keypoints(kps, name="kps").astype(float, copy=True)
    frame_indices = np.arange(keypoints.shape[0])

    for joint_index in range(keypoints.shape[1]):
        for channel_index in range(keypoints.shape[2]):
            values = keypoints[:, joint_index, channel_index]
            valid = ~np.isnan(values)
            if valid.all() or not valid.any():
                continue
            keypoints[:, joint_index, channel_index] = np.interp(
                frame_indices,
                frame_indices[valid],
                values[valid],
            )

    return keypoints


def resample_keypoints_1d(x, T_target):
    values = np.asarray(x)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("x must be a non-empty one-dimensional array")
    if T_target <= 0:
        raise ValueError("T_target must be positive")

    source_indices = np.arange(values.shape[0])
    target_indices = np.linspace(0, values.shape[0] - 1, T_target)
    return np.interp(target_indices, source_indices, values)


def resample_keypoints_2d(kps, T_target):
    keypoints = _validate_keypoints(kps, name="kps")
    if T_target <= 0:
        raise ValueError("T_target must be positive")
    if keypoints.shape[0] == T_target:
        return keypoints.copy()

    output = np.empty(
        (T_target, keypoints.shape[1], keypoints.shape[2]),
        dtype=float,
    )
    for joint_index in range(keypoints.shape[1]):
        for channel_index in range(keypoints.shape[2]):
            output[:, joint_index, channel_index] = resample_keypoints_1d(
                keypoints[:, joint_index, channel_index],
                T_target,
            )
    return output


def extract_motion_beat(keypoints, beat_dim, target_length, fps_keypoints, duration):
    keypoints = _validate_keypoints(keypoints)
    if beat_dim <= 0 or target_length <= 0:
        raise ValueError("beat_dim and target_length must be positive")
    if fps_keypoints <= 0 or duration <= 0:
        raise ValueError("fps_keypoints and duration must be positive")

    frame_velocity = np.mean(
        np.linalg.norm(keypoints[1:] - keypoints[:-1], axis=2),
        axis=1,
    )
    kinetic_velocity = np.zeros(keypoints.shape[0], dtype=float)
    valid_length = min(frame_velocity.size, kinetic_velocity.size - 1)
    kinetic_velocity[1 : valid_length + 1] = np.nan_to_num(
        frame_velocity[:valid_length],
        nan=0.0,
    )

    maximum = kinetic_velocity.max()
    if maximum > 0:
        kinetic_velocity /= maximum
    kinetic_velocity = np.nan_to_num(gaussian_filter(kinetic_velocity, sigma=2))

    motion_beats = argrelextrema(kinetic_velocity, np.less)[0]
    motion_beats = motion_beats[motion_beats < keypoints.shape[0]]

    flattened_length = target_length * beat_dim
    beat_times = motion_beats / fps_keypoints
    target_fps = flattened_length / duration
    beat_indices = np.unique(np.round(beat_times * target_fps).astype(int))
    beat_indices = beat_indices[beat_indices < flattened_length]

    beat_vector = np.zeros(flattened_length, dtype=np.int64)
    beat_vector[beat_indices] = 1
    return torch.from_numpy(beat_vector).view(target_length, beat_dim)


def make_cam(x, img_shape):
    """Normalize image coordinates to the camera coordinate range [-1, 1]."""
    height, width = img_shape
    if height <= 0 or width <= 0:
        raise ValueError("img_shape dimensions must be positive")
    scale = max(height, width)
    return np.asarray(x) / scale * 2 - 1


def coco2h36m(x):
    """Convert COCO's 17-joint ordering to the Human3.6M ordering."""
    source = np.asarray(x)
    if source.ndim < 3 or source.shape[-2] != NUM_JOINTS:
        raise ValueError(f"x must contain {NUM_JOINTS} joints on its penultimate axis")

    output = np.zeros_like(source)
    output[..., 0, :] = (source[..., 11, :] + source[..., 12, :]) * 0.5
    output[..., 1, :] = source[..., 12, :]
    output[..., 2, :] = source[..., 14, :]
    output[..., 3, :] = source[..., 16, :]
    output[..., 4, :] = source[..., 11, :]
    output[..., 5, :] = source[..., 13, :]
    output[..., 6, :] = source[..., 15, :]
    output[..., 8, :] = (source[..., 5, :] + source[..., 6, :]) * 0.5
    output[..., 7, :] = (output[..., 0, :] + output[..., 8, :]) * 0.5
    output[..., 9, :] = source[..., 0, :]
    output[..., 10, :] = (source[..., 1, :] + source[..., 2, :]) * 0.5
    output[..., 11, :] = source[..., 5, :]
    output[..., 12, :] = source[..., 7, :]
    output[..., 13, :] = source[..., 9, :]
    output[..., 14, :] = source[..., 6, :]
    output[..., 15, :] = source[..., 8, :]
    output[..., 16, :] = source[..., 10, :]
    return output


def crop_scale(motion, scale_range=(1, 1)):
    """Crop and normalize valid motion coordinates to [-1, 1]."""
    if len(scale_range) != 2 or scale_range[0] > scale_range[1]:
        raise ValueError("scale_range must be an ordered pair")

    source = np.asarray(motion)
    if source.ndim < 3 or source.shape[-1] < NUM_KEYPOINT_CHANNELS:
        raise ValueError("motion must have coordinate and confidence channels")

    valid_coordinates = source[source[..., 2] != 0][..., :2]
    if len(valid_coordinates) < 4:
        return np.zeros_like(source)

    minimum = valid_coordinates.min(axis=0)
    maximum = valid_coordinates.max(axis=0)
    ratio = np.random.uniform(scale_range[0], scale_range[1])
    scale = max(maximum - minimum) * ratio
    if scale == 0:
        return np.zeros_like(source)

    origin = (minimum + maximum - scale) / 2
    output = source.copy()
    output[..., :2] = (source[..., :2] - origin) / scale
    output[..., :2] = (output[..., :2] - 0.5) * 2
    return np.clip(output, -1, 1)
