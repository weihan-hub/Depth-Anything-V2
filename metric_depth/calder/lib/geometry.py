"""Pose / intrinsics helpers shared by the manifest + dataset builders.

Lifted out of app/build_manifest.py so both that script and the finetune-dataset
builder (lib/shard.py) import a single implementation.
"""
import math


def axis_angle_to_matrix(ax, ay, az, angle_deg):
    """(unit-axis, angle_degrees) -> 3x3 rotation matrix (Rodrigues)."""
    theta = math.radians(angle_deg)
    # The stored (x, y, z) is a unit axis; renormalize defensively.
    n = math.sqrt(ax * ax + ay * ay + az * az)
    if n < 1e-12:
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    ax, ay, az = ax / n, ay / n, az / n
    c, s = math.cos(theta), math.sin(theta)
    C = 1.0 - c
    return [
        [c + ax * ax * C,      ax * ay * C - az * s, ax * az * C + ay * s],
        [ay * ax * C + az * s, c + ay * ay * C,      ay * az * C - ax * s],
        [az * ax * C - ay * s, az * ay * C + ax * s, c + az * az * C],
    ]


def pose_to_4x4(pose):
    """{'axis_angle': {x,y,z,angle_degrees}, 'translation': {x,y,z}} -> 4x4 list."""
    aa = pose["axis_angle"]
    t = pose["translation"]
    R = axis_angle_to_matrix(aa["x"], aa["y"], aa["z"], aa["angle_degrees"])
    return [
        [R[0][0], R[0][1], R[0][2], t["x"]],
        [R[1][0], R[1][1], R[1][2], t["y"]],
        [R[2][0], R[2][1], R[2][2], t["z"]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def projection_to_K(data):
    """3x4 row-major projection matrix data -> 3x3 K.

    data = [fx, 0, cx, 0,  0, fy, cy, 0,  0, 0, 1, 0]
    """
    fx, _, cx = data[0], data[1], data[2]
    fy, cy = data[5], data[6]
    return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]


def rig_and_side(camera_name):
    """'stereo_camera_front_left' -> ('front', 'left')."""
    parts = camera_name.split("_")
    side = parts[-1]               # left / right (stereo eye)
    rig = parts[-2]                # front / left / right (rig position)
    return rig, side
