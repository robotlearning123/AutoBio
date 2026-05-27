"""Torch-based IK solvers for the ALOHA robot arm.

Ports the analytical IK from ``AutoBio/autobio/aloha_analytical_ik.py``
from numpy to torch for batched execution in Isaac Lab.
"""

from __future__ import annotations

import torch
import numpy as np


def _close_to_zero(x: torch.Tensor, tol: float = 1e-8) -> torch.Tensor:
    return torch.abs(x) < tol


def site_pose_to_wrist_pos(
    site_pos: torch.Tensor, site_quat: torch.Tensor
) -> torch.Tensor:
    """Convert site pose to wrist position using spherical wrist property.

    Args:
        site_pos: (..., 3) position.
        site_quat: (..., 4) quaternion (wxyz).

    Returns:
        (..., 3) wrist position.
    """
    # Local axis and length for left/gripper site
    local_axis = torch.tensor([0.9998872305243953, 0.0, -0.015017530897414616],
                              device=site_pos.device, dtype=site_pos.dtype)
    length = 0.1997665275665571

    # Rotate local axis by site quaternion
    axis = _quat_apply(site_quat, local_axis)
    return site_pos - axis * length


def solve_base(wrist_pos: torch.Tensor) -> torch.Tensor | None:
    """Solve base joint angles (q0, q1, q2) given wrist position.

    Args:
        wrist_pos: (3,) wrist position.

    Returns:
        (N, 3) solutions for (q0, q1, q2), or None if no solution.
    """
    a, b, c = 0.3, 0.05955, 0.12705
    x, y, z = wrist_pos[0], wrist_pos[1], wrist_pos[2]
    v = z - c

    if _close_to_zero(x) and _close_to_zero(y):
        # Wrist upright - q0 indeterminate
        return None

    q0 = torch.atan2(y, x)
    u = torch.sqrt(x ** 2 + y ** 2)

    slns = []
    q12 = _solve_planar_2link(u, v)
    if q12 is not None:
        slns.append(torch.cat([q0.expand(q12.shape[0], 1), q12], dim=1))
    q12 = _solve_planar_2link(-u, v)
    if q12 is not None:
        slns.append(torch.cat([(q0 + np.pi).expand(q12.shape[0], 1), q12], dim=1))

    if not slns:
        return None
    return torch.cat(slns, dim=0)


def _solve_planar_2link(u: float, v: float) -> torch.Tensor | None:
    """Solve planar 2-link IK for (q1, q2)."""
    a, b = 0.3, 0.05955

    A = -2 * a * u + 2 * b * v
    B = -2 * a * v - 2 * b * u
    C = b ** 2 + u ** 2 + v ** 2

    q1 = _solve_linear_trig(A, B, C)
    if q1 is None:
        return None

    cos_q12 = (u - a * torch.sin(q1) - b * torch.cos(q1)) / a
    sin_q12 = -(v - a * torch.cos(q1) + b * torch.sin(q1)) / a
    q2 = torch.atan2(sin_q12, cos_q12) - q1

    return torch.stack([q1, q2], dim=-1)


def _solve_linear_trig(A: float, B: float, C: float) -> torch.Tensor | None:
    """Solve A*sin(x) + B*cos(x) + C = 0."""
    R = torch.sqrt(torch.tensor(A ** 2 + B ** 2))
    theta = torch.atan2(torch.tensor(B), torch.tensor(A))
    t = -C / R

    if torch.abs(t) > 1:
        return None

    inv = torch.asin(t)
    return torch.tensor([inv - theta, np.pi - inv - theta])


def solve_wrist(target_quat: torch.Tensor) -> torch.Tensor:
    """Solve wrist joint angles (q3, q4, q5) given target quaternion in wrist frame.

    Args:
        target_quat: (4,) quaternion (wxyz) in wrist frame.

    Returns:
        (N, 3) solutions for (q3, q4, q5).
    """
    w, x, y, z = target_quat[0], target_quat[1], target_quat[2], target_quat[3]

    c4_sqr = w ** 2 + x ** 2
    p35 = torch.atan2(x, w)

    if _close_to_zero(1 - c4_sqr):
        # Gimbal lock
        return torch.tensor([[0.0, 0.0, p35 * 2]])

    q4 = torch.acos(torch.clamp(2 * c4_sqr - 1, -1.0, 1.0))
    m35 = torch.atan2(z, y)

    return torch.tensor([
        [p35 + m35, q4, p35 - m35],
        [p35 + m35 + np.pi, -q4, p35 - m35 - np.pi],
    ])


def prune_by_bounds(sln: torch.Tensor) -> torch.Tensor:
    """Remove solutions outside joint limits."""
    bounds = torch.tensor([
        [-3.14158,  3.14158],
        [-1.85005,  1.25664],
        [-1.76278,  1.6057],
        [-3.14158,  3.14158],
        [-1.8675,   2.23402],
        [-6.28,     6.28],
    ], device=sln.device, dtype=sln.dtype)

    # Normalize to [-pi, pi]
    sln = (sln + np.pi) % (2 * np.pi) - np.pi
    mask = torch.all((sln >= bounds[:, 0]) & (sln <= bounds[:, 1]), dim=1)
    return sln[mask]


def aloha_analytical_ik(
    target_pos: torch.Tensor, target_quat: torch.Tensor
) -> torch.Tensor | None:
    """Compute analytical IK for the ALOHA arm.

    Args:
        target_pos: (3,) target end-effector position.
        target_quat: (4,) target end-effector quaternion (wxyz).

    Returns:
        (N, 6) joint angle solutions, or None if no solution.
    """
    wrist_pos = site_pose_to_wrist_pos(target_pos, target_quat)
    q_base = solve_base(wrist_pos)
    if q_base is None:
        return None

    # Solve wrist for each base solution
    slns = []
    for q in q_base:
        q0, q1, q2 = q[0], q[1], q[2]
        # Wrist FK
        c0 = torch.cos(q0 / 2)
        s0 = torch.sin(q0 / 2)
        c12 = torch.cos(q1 / 2 + q2 / 2)
        s12 = torch.sin(q1 / 2 + q2 / 2)
        wrist_quat = torch.stack([c0 * c12, -s0 * s12, c0 * s12, s0 * c12])

        # Target in wrist frame
        target_local = _quat_multiply(_quat_inv(wrist_quat), target_quat)
        q_wrist = solve_wrist(target_local)

        for ws in q_wrist:
            full = torch.cat([q, ws])
            slns.append(full)

    if not slns:
        return None

    sln = torch.stack(slns)
    return prune_by_bounds(sln)


# --- Quaternion utilities ---

def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply quaternion rotation to a vector. quat: (..., 4) wxyz, vec: (..., 3)."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    vx, vy, vz = vec[..., 0], vec[..., 1], vec[..., 2]

    t2 = w * x
    t3 = w * y
    t4 = w * z
    t5 = -x * x
    t6 = x * y
    t7 = x * z
    t8 = -y * y
    t9 = y * z
    t10 = -z * z

    return torch.stack([
        2 * (t8 + t10) * vx + 2 * (t6 - t4) * vy + 2 * (t7 + t3) * vz + vx,
        2 * (t6 + t4) * vx + 2 * (t5 + t10) * vy + 2 * (t9 - t2) * vz + vy,
        2 * (t7 - t3) * vx + 2 * (t9 + t2) * vy + 2 * (t5 + t8) * vz + vz,
    ], dim=-1)


def _quat_inv(quat: torch.Tensor) -> torch.Tensor:
    """Quaternion inverse (conjugate for unit quaternions). quat: (..., 4) wxyz."""
    return torch.stack([quat[..., 0], -quat[..., 1], -quat[..., 2], -quat[..., 3]], dim=-1)


def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product. q1, q2: (..., 4) wxyz."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def dls_ik(
    jacobian: torch.Tensor,
    delta_pos: torch.Tensor,
    damping: float = 0.05,
) -> torch.Tensor:
    """Damped Least Squares IK solver.

    Args:
        jacobian: (..., 6, N) Jacobian matrix.
        delta_pos: (..., 6) desired twist (linear + angular).
        damping: Damping factor.

    Returns:
        (..., N) joint angle increments.
    """
    J = jacobian
    Jt = J.transpose(-1, -2)
    JJt = J @ Jt
    damping_sq = damping ** 2 * torch.eye(JJt.shape[-1], device=JJt.device, dtype=JJt.dtype)
    return Jt @ torch.linalg.solve(JJt + damping_sq, delta_pos)
