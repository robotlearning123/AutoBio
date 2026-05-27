"""Physics approximation utilities for AutoBio tasks.

Provides approximations for MuJoCo-specific physics features that
cannot be directly replicated in PhysX:
- Screw thread helical coupling
- Detent mechanism spring forces
- Liquid surface containment
"""

from __future__ import annotations

import torch


def screw_thread_force(
    revolute_angle: torch.Tensor,
    prismatic_pos: torch.Tensor,
    pitch: float = 0.003,
    stiffness: float = 1000.0,
) -> torch.Tensor:
    """Compute spring force coupling revolute and prismatic joints for screw thread.

    The helical relationship: ``dz = pitch * dtheta / (2 * pi)``

    Args:
        revolute_angle: (..., 1) current revolute joint angle.
        prismatic_pos: (..., 1) current prismatic joint position.
        pitch: Thread pitch in meters per revolution.
        stiffness: Spring stiffness (N/m).

    Returns:
        (..., 1) force to apply to the prismatic joint.
    """
    target_z = pitch * revolute_angle / (2 * torch.pi)
    error = target_z - prismatic_pos
    return stiffness * error


def detent_force(
    angle: torch.Tensor,
    detent_positions: torch.Tensor,
    stiffness: float = 50.0,
    width: float = 0.1,
) -> torch.Tensor:
    """Compute detent snap force toward nearest detent position.

    Args:
        angle: (..., 1) current joint angle.
        detent_positions: (N,) detent positions.
        stiffness: Detent spring stiffness.
        width: Width of detent catch region.

    Returns:
        (..., 1) torque to apply.
    """
    # Find nearest detent for each angle
    distances = torch.abs(angle.unsqueeze(-1) - detent_positions)
    nearest_idx = torch.argmin(distances, dim=-1)
    nearest = detent_positions[nearest_idx]

    error = nearest - angle
    # Only apply force within catch region
    in_range = torch.abs(error) < width
    return stiffness * error * in_range.float()


def liquid_surface_check(
    tip_pos: torch.Tensor,
    container_pos: torch.Tensor,
    surface_height: float = 0.05,
) -> torch.Tensor:
    """Check if pipette tip is below liquid surface.

    Args:
        tip_pos: (..., 3) pipette tip position.
        container_pos: (..., 3) container position.
        surface_height: Liquid surface height relative to container.

    Returns:
        (..., 1) boolean tensor, True if tip is below surface.
    """
    liquid_z = container_pos[..., 2] + surface_height
    return (tip_pos[..., 2] < liquid_z).unsqueeze(-1)
