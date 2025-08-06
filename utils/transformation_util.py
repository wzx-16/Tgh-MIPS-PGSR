# Copyright (c) Meta Platforms, Inc. and affiliates.
# https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion

import torch
import torch.nn.functional as F
import numpy as np

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

def axis_angle_to_euler_angles(axis_angle: torch.Tensor) -> torch.Tensor:
    return matrix_to_euler_angles(axis_angle_to_matrix(axis_angle))

def euler_angles_to_axis_angle(euler_angles: torch.Tensor) -> torch.Tensor:
    return matrix_to_axis_angle(euler_angles_to_matrix(euler_angles))

def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles],
        dim=-1,
    )
    return quaternions


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    # print(r.shape, i.shape, j.shape, k.shape)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack(
                [q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1
            ),
            torch.stack(
                [m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1
            ),
            torch.stack(
                [m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1
            ),
            torch.stack(
                [m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1
            ),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        # pyre-ignore[16]
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5,
        :,
    ].reshape(batch_dim + (4,))
    
def matrix_to_euler_angles(R):
    sy = torch.sqrt(R[..., 0, 0] * R[..., 0, 0] + R[..., 1, 0] * R[..., 1, 0])
    singular = sy < 1e-6
    # print(singular.shape)
    x = torch.zeros_like(sy)
    y = torch.zeros_like(sy)
    z = torch.zeros_like(sy)
    # if not singular:
    #     x = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    #     y = torch.atan2(-R[..., 2, 0], sy)
    #     z = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    # else:
    #     x = torch.atan2(-R[..., 1, 2], R[..., 1, 1])
    #     y = torch.atan2(-R[..., 2, 0], sy)
    #     z = 0

    x[singular] = torch.atan2(-R[singular, 1, 2], R[singular, 1, 1])
    y[singular] = torch.atan2(-R[singular, 2, 0], sy[singular])
    
    x[~singular] = torch.atan2(R[~singular, 2, 1], R[~singular, 2, 2])
    y[~singular] = torch.atan2(-R[~singular, 2, 0], sy[~singular])
    z[~singular] = torch.atan2(R[~singular, 1, 0], R[~singular, 0, 0])
    return torch.stack([x, y, z], dim=-1)

    # return torch.tensor([x, y, z]).to(dtype=R.dtype, device=R.device)
    
def euler_angles_to_matrix(euler_angles):
    """
    Convert Euler angles (in radians) to a rotation matrix in PyTorch.
    Input:
        euler_angles: Tensor of shape (batch_size, 3) representing the Euler angles in radians.
    Output:
        rotation_matrix: Tensor of shape (batch_size, 3, 3) representing the rotation matrix.
    """
    # Extract Euler angles
    roll = euler_angles[..., 0]
    pitch = euler_angles[..., 1]
    yaw = euler_angles[..., 2]
    
    # Compute sin and cos values
    cos_r, sin_r = torch.cos(roll), torch.sin(roll)
    cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
    
    # Compute the rotation matrix
    rotation_matrix = torch.stack([
        torch.stack([cos_p*cos_y, cos_y*sin_r*sin_p - cos_r*sin_y, cos_r*cos_y*sin_p + sin_r*sin_y], dim=-1),
        torch.stack([cos_p*sin_y, cos_r*cos_y + sin_r*sin_p*sin_y, -cos_y*sin_r + cos_r*sin_p*sin_y], dim=-1),
        torch.stack([-sin_p, cos_p*sin_r, cos_r*cos_p], dim=-1)
    ], dim=-2)
    
    return rotation_matrix
    
# For numpy 
    
def rotation_6d_to_matrix_np(d6: np.ndarray) -> np.ndarray:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1/np.maximum(np.linalg.norm(a1, ord=2, axis = -1, keepdims=True), 1e-12)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2/np.maximum(np.linalg.norm(b2, ord=2, axis = -1, keepdims=True), 1e-12)
    b3 = np.linalg.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2)


def matrix_to_rotation_6d_np(matrix: np.ndarray) -> np.ndarray:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.shape[:-2]
    return matrix[..., :2, :].copy().reshape(batch_dim + (6,))


def quaternion_to_axis_angle_np(quaternions: np.ndarray) -> np.ndarray:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = np.linalg.norm(quaternions[..., 1:], ord=2, axis = -1, keepdims=True)
    half_angles = np.arctan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = np.absolute(angles) < eps
    sin_half_angles_over_angles = np.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        np.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def matrix_to_axis_angle_np(matrix: np.ndarray) -> np.ndarray:
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle_np(matrix_to_quaternion_np(matrix))


def axis_angle_to_matrix_np(axis_angle: np.ndarray) -> np.ndarray:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix_np(axis_angle_to_quaternion_np(axis_angle))


def axis_angle_to_quaternion_np(axis_angle: np.ndarray) -> np.ndarray:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = np.linalg.norm(axis_angle, ord=2, axis = -1, keepdims=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = np.absolute(angles) < eps
    sin_half_angles_over_angles = np.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        np.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = np.concatenate(
        [np.cos(half_angles), axis_angle * sin_half_angles_over_angles],
        axis=-1,
    )
    return quaternions


def quaternion_to_matrix_np(quaternions: np.ndarray) -> np.ndarray:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    # r, i, j, k = np.split(quaternions, quaternions.shape[-1], axis=-1)
    r, i, j, k = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]
    # print(r.shape, i.shape, j.shape, k.shape)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = np.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _sqrt_positive_part_np(x: np.ndarray) -> np.ndarray:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = np.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = np.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion_np(matrix: np.ndarray) -> np.ndarray:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    # m00, m01, m02, m10, m11, m12, m20, m21, m22 = np.split(
    #     matrix.reshape(batch_dim + (9,)), 9, axis=-1
    # )
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2], matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2], matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    
    q_abs = _sqrt_positive_part_np(
        np.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            axis=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = np.stack(
        [
            np.stack(
                [q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], axis=-1
            ),
            np.stack(
                [m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], axis=-1
            ),
            np.stack(
                [m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], axis=-1
            ),
            np.stack(
                [m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], axis=-1
            ),
        ],
        axis=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = np.array(0.1, dtype=q_abs.dtype)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].clip(min=flr))
    

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        # pyre-ignore[16]
        one_hot_encode_np(q_abs.argmax(axis=-1), num_classes=4) > 0.5,
        :,
    ].reshape(batch_dim + (4,))
    
def one_hot_encode_np(indices, num_classes):
    one_hot = np.zeros((len(indices), num_classes))
    one_hot[np.arange(len(indices)), indices] = 1
    return one_hot

def matrix_to_euler_angles_np(R):
    sy = np.sqrt(R[..., 0, 0] * R[..., 0, 0] + R[..., 1, 0] * R[..., 1, 0])
    singular = sy < 1e-6
    x = np.zeros_like(sy)
    y = np.zeros_like(sy)
    z = np.zeros_like(sy)
    x[singular] = np.arctan2(-R[singular, 1, 2], R[singular, 1, 1])
    y[singular] = np.arctan2(-R[singular, 2, 0], sy[singular])
    
    x[~singular] = np.arctan2(R[~singular, 2, 1], R[~singular, 2, 2])
    y[~singular] = np.arctan2(-R[~singular, 2, 0], sy[~singular])
    z[~singular] = np.arctan2(R[~singular, 1, 0], R[~singular, 0, 0])
    return np.stack([x, y, z], axis=-1)

def axis_angle_to_euler_angles_np(axis_angle: np.ndarray) -> np.ndarray:
    return matrix_to_euler_angles_np(axis_angle_to_matrix_np(axis_angle))

def euler_angles_to_matrix_np(euler_angles):
    """
    Convert Euler angles (in radians) to a rotation matrix in PyTorch.
    Input:
        euler_angles: Tensor of shape (batch_size, 3) representing the Euler angles in radians.
    Output:
        rotation_matrix: Tensor of shape (batch_size, 3, 3) representing the rotation matrix.
    """
    # Extract Euler angles
    roll = euler_angles[..., 0]
    pitch = euler_angles[..., 1]
    yaw = euler_angles[..., 2]
    
    # Compute sin and cos values
    cos_r, sin_r = np.cos(roll), np.sin(roll)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    
    # Compute the rotation matrix
    rotation_matrix = np.stack([
        np.stack([cos_p*cos_y, cos_y*sin_r*sin_p - cos_r*sin_y, cos_r*cos_y*sin_p + sin_r*sin_y], axis=-1),
        np.stack([cos_p*sin_y, cos_r*cos_y + sin_r*sin_p*sin_y, -cos_y*sin_r + cos_r*sin_p*sin_y], axis=-1),
        np.stack([-sin_p, cos_p*sin_r, cos_r*cos_p], axis=-1)
    ], axis=-2)
    
    return rotation_matrix

def euler_angles_to_axis_angle_np(euler_angles: np.ndarray) -> np.ndarray:
    return matrix_to_axis_angle_np(euler_angles_to_matrix_np(euler_angles))


if __name__ == "__main__":
    a = np.load('/media/bbnc/ssd/lenovo/lyg9/smpl_params.npz')
    a = dict(a)
    for ii in range(0, 2700, 5):
        b = torch.from_numpy(a['jaw_pose'][ii])
        b = matrix_to_euler_angles(axis_angle_to_matrix(b))
        if b[0] > 0.02:
            print(b, ii)