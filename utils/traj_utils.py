import numpy as np
from utils.graphics_utils import getProjectionMatrix

def _new_poses_pan(poses, n_frames, pan_angle_deg=60.0):
    base = poses[len(poses) // 2]
    R0, t0 = base[:3, :3], base[:3, 3]
    angles = np.linspace(-np.radians(pan_angle_deg/2), np.radians(pan_angle_deg/2), n_frames)
    return np.stack([
        np.concatenate([_axis_angle_rotation(R0[:, 1], a) @ R0, t0[:, None]], axis=-1)
        for a in angles
    ])

def _new_poses_tilt(poses, n_frames, tilt_angle_deg=30.0):
    base = poses[len(poses) // 2]
    R0, t0 = base[:3, :3], base[:3, 3]
    angles = np.linspace(-np.radians(tilt_angle_deg/2), np.radians(tilt_angle_deg/2), n_frames)
    return np.stack([
        np.concatenate([_axis_angle_rotation(R0[:, 0], a) @ R0, t0[:, None]], axis=-1)
        for a in angles
    ])

def _new_poses_dolly(poses, n_frames, dolly_factor=0.2):
    base = poses[len(poses) // 2]
    R0, t0 = base[:3, :3], base[:3, 3]
    depth_range = np.linalg.norm(poses[:, :3, 3].max(0) - poses[:, :3, 3].min(0))
    offsets = np.linspace(0, depth_range * dolly_factor, n_frames)
    return np.stack([
        np.concatenate([R0, (t0 - R0[:, 2] * d)[:, None]], axis=-1)
        for d in offsets
    ])

def _new_poses_truck(poses, n_frames, truck_factor=0.3):
    base = poses[len(poses) // 2]
    R0, t0 = base[:3, :3], base[:3, 3]
    scene_width = np.linalg.norm(poses[:, :3, 3].max(0) - poses[:, :3, 3].min(0))
    offsets = np.linspace(-scene_width * truck_factor / 2, scene_width * truck_factor / 2, n_frames)
    return np.stack([
        np.concatenate([R0, (t0 + R0[:, 0] * d)[:, None]], axis=-1)
        for d in offsets
    ])

def _new_poses_pedestal(poses, n_frames, pedestal_factor=0.3):
    base = poses[len(poses) // 2]
    R0, t0 = base[:3, :3], base[:3, 3]
    scene_height = np.linalg.norm(poses[:, :3, 3].max(0) - poses[:, :3, 3].min(0))
    offsets = np.linspace(-scene_height * pedestal_factor / 2, scene_height * pedestal_factor / 2, n_frames)
    return np.stack([
        np.concatenate([R0, (t0 + R0[:, 1] * d)[:, None]], axis=-1)
        for d in offsets
    ])


def _new_poses_zoom(poses, n_frames):
    """Repeat the base pose — position/orientation stays fixed; zoom is applied separately."""
    base = poses[len(poses) // 2]
    return np.stack([base] * n_frames)


def _apply_zoom(traj, zoom_factor=0.5):
    """
    Animate FoV from wide to narrow (zoom in) across the trajectory.
    zoom_factor: fraction to scale tan(FoV/2) by at the end (< 1 = zoom in, > 1 = zoom out)
    """
    n = len(traj)
    fov0x = traj[0].FoVx
    fov0y = traj[0].FoVy

    for i, cam in enumerate(traj):
        t = i / max(n - 1, 1)
        scale = 1.0 + t * (zoom_factor - 1.0)
        cam.FoVx = 2 * np.arctan(np.tan(fov0x / 2) * scale)
        cam.FoVy = 2 * np.arctan(np.tan(fov0y / 2) * scale)
        cam.projection_matrix = getProjectionMatrix(
            znear=cam.znear, zfar=cam.zfar, fovX=cam.FoVx, fovY=cam.FoVy
        ).transpose(0, 1).cuda()
        cam.full_proj_transform = (
            cam.world_view_transform.unsqueeze(0)
            .bmm(cam.projection_matrix.unsqueeze(0))
            .squeeze(0)
        )

    return traj


# ─────────────────────────────────────────────
# utility: axis-angle rotation matrix
# ─────────────────────────────────────────────

def _axis_angle_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation matrix for rotating *angle_rad* around *axis*."""
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    t    = 1 - c
    x, y, z = axis
    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ],
    ])