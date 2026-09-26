"""Turret-frame geometry.

Azimuth 0 is pan home. Positive azimuth is to the right (clockwise from above).
Elevation 0 is horizontal. Positive elevation is up. Angles are degrees.
range_m is never estimated.

Intrinsics come from calibration when fx, fy, cx, and cy are all set.
Otherwise they are derived from the frame size and the configured field of view:

    fx = (W / 2) / tan(hfov / 2)
    fy = (H / 2) / tan(vfov / 2)
    cx = W / 2
    cy = H / 2

A pixel (u, v) becomes a camera ray, x right, y up, z forward:

    d = [(u - cx) / fx, -(v - cy) / fy, 1]

That ray is rotated by tilt about x, then by pan about vertical.
The full rotation is used. There is no small-angle shortcut.

    az = atan2(dx, dz)
    el = atan2(dy, sqrt(dx^2 + dz^2))
"""

from __future__ import annotations

import math

from vision.config import Settings


def intrinsics(width: int, height: int, settings: Settings) -> tuple[float, float, float, float]:
    """Return fx, fy, cx, cy in pixels of this frame."""
    if None not in (settings.fx, settings.fy, settings.cx, settings.cy):
        return float(settings.fx), float(settings.fy), float(settings.cx), float(settings.cy)
    hfov = math.radians(settings.hfov_deg)
    vfov = math.radians(settings.vfov_deg)
    fx = (width / 2) / math.tan(hfov / 2)
    fy = (height / 2) / math.tan(vfov / 2)
    return fx, fy, width / 2, height / 2


def pixel_to_azimuth_elevation(
    u: float,
    v: float,
    width: int,
    height: int,
    pan_deg: float,
    tilt_deg: float,
    settings: Settings,
) -> tuple[float, float]:
    """Point the laser at the object whose box center is (u, v)."""
    fx, fy, cx, cy = intrinsics(width, height, settings)
    ray = ((u - cx) / fx, -((v - cy) / fy), 1.0)
    x, y, z = camera_ray_to_world(ray, pan_deg, tilt_deg)
    azimuth = math.degrees(math.atan2(x, z))
    elevation = math.degrees(math.atan2(y, math.hypot(x, z)))
    return azimuth, elevation


def box_center_angles(
    bbox_px: list[int] | tuple[int, int, int, int],
    width: int,
    height: int,
    pan_deg: float,
    tilt_deg: float,
    settings: Settings,
) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox_px
    return pixel_to_azimuth_elevation(
        (x1 + x2) / 2,
        (y1 + y2) / 2,
        width,
        height,
        pan_deg,
        tilt_deg,
        settings,
    )


def direction_to_pixel(
    azimuth_deg: float,
    elevation_deg: float,
    pan_deg: float,
    tilt_deg: float,
    width: int,
    height: int,
    settings: Settings,
) -> tuple[float, float, float]:
    """Inverse of pixel_to_azimuth_elevation.

    Returns (u, v, camera_z). camera_z <= 0 means the direction is behind the camera.
    """
    fx, fy, cx, cy = intrinsics(width, height, settings)
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    world = (
        math.sin(azimuth) * math.cos(elevation),
        math.sin(elevation),
        math.cos(azimuth) * math.cos(elevation),
    )
    x, y, z = world_to_camera(world, pan_deg, tilt_deg)
    if z == 0:
        return cx, cy, 0.0
    u = cx + fx * (x / z)
    v = cy - fy * (y / z)
    return u, v, z


def angular_distance_deg(azimuth_a: float, elevation_a: float, azimuth_b: float, elevation_b: float) -> float:
    """Angle between two turret directions, in degrees."""
    first = _direction(azimuth_a, elevation_a)
    second = _direction(azimuth_b, elevation_b)
    dot = first[0] * second[0] + first[1] * second[1] + first[2] * second[2]
    if dot >= 1.0 - 1e-12:
        return 0.0
    if dot <= -1.0:
        return 180.0
    return math.degrees(math.acos(max(-1.0, dot)))


def camera_ray_to_world(
    ray: tuple[float, float, float], pan_deg: float, tilt_deg: float
) -> tuple[float, float, float]:
    """Tilt about x (positive tilt looks up), then pan about vertical (positive pan looks right)."""
    x, y, z = ray
    tilt = math.radians(tilt_deg)
    pan = math.radians(pan_deg)
    cos_t, sin_t = math.cos(tilt), math.sin(tilt)
    # R_x(-tilt): positive tilt raises the optical axis.
    y1 = y * cos_t + z * sin_t
    z1 = -y * sin_t + z * cos_t
    cos_p, sin_p = math.cos(pan), math.sin(pan)
    # R_y(pan): positive pan turns +z toward +x.
    x2 = x * cos_p + z1 * sin_p
    y2 = y1
    z2 = -x * sin_p + z1 * cos_p
    return x2, y2, z2


def world_to_camera(
    world: tuple[float, float, float], pan_deg: float, tilt_deg: float
) -> tuple[float, float, float]:
    """Inverse of camera_ray_to_world."""
    x, y, z = world
    pan = math.radians(pan_deg)
    tilt = math.radians(tilt_deg)
    cos_p, sin_p = math.cos(pan), math.sin(pan)
    x1 = x * cos_p - z * sin_p
    y1 = y
    z1 = x * sin_p + z * cos_p
    cos_t, sin_t = math.cos(tilt), math.sin(tilt)
    y2 = y1 * cos_t - z1 * sin_t
    z2 = y1 * sin_t + z1 * cos_t
    return x1, y2, z2


def _direction(azimuth_deg: float, elevation_deg: float) -> tuple[float, float, float]:
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    return (
        math.sin(azimuth) * math.cos(elevation),
        math.sin(elevation),
        math.cos(azimuth) * math.cos(elevation),
    )
