"""Torch-based trajectory interpolation utilities.

Ports the trajectory helpers from AutoBio's ``kinematics.py`` and ``topp.py``
to torch for batched execution.
"""

from __future__ import annotations

import torch
import numpy as np


def slerp(q0: torch.Tensor, q1: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation between two quaternions.

    Args:
        q0: (4,) start quaternion (wxyz).
        q1: (4,) end quaternion (wxyz).
        t: Interpolation parameter in [0, 1].

    Returns:
        (4,) interpolated quaternion.
    """
    dot = torch.dot(q0, q1)
    # Ensure shortest path
    if dot < 0:
        q1 = -q1
        dot = -dot

    dot = torch.clamp(dot, -1.0, 1.0)

    if dot > 0.9995:
        # Linear interpolation for very close quaternions
        result = q0 + t * (q1 - q0)
        return result / torch.norm(result)

    theta_0 = torch.acos(dot)
    theta = theta_0 * t
    sin_theta = torch.sin(theta)
    sin_theta_0 = torch.sin(theta_0)

    s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    result = s0 * q0 + s1 * q1
    return result / torch.norm(result)


def linear_interpolate(
    start: torch.Tensor, end: torch.Tensor, num_steps: int
) -> list[torch.Tensor]:
    """Linear interpolation between two tensors.

    Args:
        start: Start value.
        end: End value.
        num_steps: Number of interpolation steps (inclusive).

    Returns:
        List of interpolated values.
    """
    points = []
    for i in range(num_steps + 1):
        t = i / num_steps
        points.append(start + t * (end - start))
    return points


def pose_interpolate(
    pos_start: torch.Tensor,
    quat_start: torch.Tensor,
    pos_end: torch.Tensor,
    quat_end: torch.Tensor,
    num_steps: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Interpolate position (linear) and orientation (slerp) simultaneously.

    Args:
        pos_start: (3,) start position.
        quat_start: (4,) start quaternion (wxyz).
        pos_end: (3,) end position.
        quat_end: (4,) end quaternion (wxyz).
        num_steps: Number of steps (inclusive).

    Returns:
        List of (pos, quat) tuples.
    """
    poses = []
    for i in range(num_steps + 1):
        t = i / num_steps
        pos = pos_start + t * (pos_end - pos_start)
        quat = slerp(quat_start, quat_end, t)
        poses.append((pos, quat))
    return poses
