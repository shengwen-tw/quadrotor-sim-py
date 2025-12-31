import numpy as np
import torch

from torch import Tensor


class TensorSE3:
    @staticmethod
    def euler_to_rotmat(roll: Tensor, pitch: Tensor, yaw: Tensor) -> Tensor:
        """Convert Euler angles to rotation matrix"""
        cos_phi = torch.cos(roll)
        cos_theta = torch.cos(pitch)
        cos_psi = torch.cos(yaw)

        sin_phi = torch.sin(roll)
        sin_theta = torch.sin(pitch)
        sin_psi = torch.sin(yaw)

        B = roll.shape[0]
        R = torch.empty((B, 3, 3), device=roll.device, dtype=roll.dtype)

        R[:, 0, 0] = cos_theta * cos_psi
        R[:, 0, 1] = -cos_phi * sin_psi + sin_phi * sin_theta * cos_psi
        R[:, 0, 2] = sin_phi * sin_psi + cos_phi * sin_theta * cos_psi

        R[:, 1, 0] = cos_theta * sin_psi
        R[:, 1, 1] = cos_phi * cos_psi + sin_phi * sin_theta * sin_psi
        R[:, 1, 2] = -sin_phi * cos_psi + cos_phi * sin_theta * sin_psi

        R[:, 2, 0] = -sin_theta
        R[:, 2, 1] = sin_phi * cos_theta
        R[:, 2, 2] = cos_phi * cos_theta

        return R

    @staticmethod
    def rotmat_to_euler(R: Tensor) -> Tensor:
        """Convert rotation matrix to Euler angles"""
        roll = torch.atan2(R[:, 2, 1], R[:, 2, 2])
        pitch = torch.asin(-R[:, 2, 0])
        yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        return torch.stack([roll, pitch, yaw], dim=1)

    @staticmethod
    def vee_map_3x3(S: Tensor) -> Tensor:
        """Convert skew-symmetric matrix to vector (vee map)"""
        return torch.stack([S[:, 2, 1], S[:, 0, 2], S[:, 1, 0]], dim=1)

    @staticmethod
    def hat_map_3x3(vec: Tensor) -> Tensor:
        """Convert vector to skew-symmetric matrix (hat map)"""
        vx = vec[:, 0]
        vy = vec[:, 1]
        vz = vec[:, 2]
        S = vec.new_zeros((vec.shape[0], 3, 3))
        S[:, 0, 1] = -vz
        S[:, 0, 2] = vy
        S[:, 1, 0] = vz
        S[:, 1, 2] = -vx
        S[:, 2, 0] = -vy
        S[:, 2, 1] = vx
        return S

    @staticmethod
    def rotmat_orthonormalize(R: Tensor) -> Tensor:
        """
        Fast rotation-matrix orthonormalization.
        Ref: "Direction Cosine Matrix IMU: Theory"
        """
        x = R[:, 0, :]
        y = R[:, 1, :]
        error = (x * y).sum(dim=1, keepdim=True)

        # Orthogonalization
        x_orth = x - 0.5 * error * y
        y_orth = y - 0.5 * error * x
        z_orth = torch.cross(x_orth, y_orth, dim=1)

        # Normalization
        x_dot = (x_orth * x_orth).sum(dim=1, keepdim=True)
        y_dot = (y_orth * y_orth).sum(dim=1, keepdim=True)
        z_dot = (z_orth * z_orth).sum(dim=1, keepdim=True)
        x_normal = 0.5 * (3.0 - x_dot) * x_orth
        y_normal = 0.5 * (3.0 - y_dot) * y_orth
        z_normal = 0.5 * (3.0 - z_dot) * z_orth

        R_out = R.new_empty(R.shape)
        R_out[:, 0, :] = x_normal
        R_out[:, 1, :] = y_normal
        R_out[:, 2, :] = z_normal
        return R_out

    @staticmethod
    def get_prv_angle(R: Tensor) -> Tensor:
        """Get principal rotation vector angle from a rotation matrix"""
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        return torch.acos(0.5 * (trace - 1.0))


class NumpySE3:
    @staticmethod
    def euler_to_rotmat(roll, pitch, yaw):
        """Convert Euler angles to rotation matrix"""
        cos_phi = np.cos(roll)
        cos_theta = np.cos(pitch)
        cos_psi = np.cos(yaw)

        sin_phi = np.sin(roll)
        sin_theta = np.sin(pitch)
        sin_psi = np.sin(yaw)

        rotmat00 = cos_theta * cos_psi
        rotmat01 = -cos_phi * sin_psi + sin_phi * sin_theta * cos_psi
        rotmat02 = sin_phi * sin_psi + cos_phi * sin_theta * cos_psi

        rotmat10 = cos_theta * sin_psi
        rotmat11 = cos_phi * cos_psi + sin_phi * sin_theta * sin_psi
        rotmat12 = -sin_phi * cos_psi + cos_phi * sin_theta * sin_psi

        rotmat20 = -sin_theta
        rotmat21 = sin_phi * cos_theta
        rotmat22 = cos_phi * cos_theta

        return np.array([
            [rotmat00, rotmat01, rotmat02],
            [rotmat10, rotmat11, rotmat12],
            [rotmat20, rotmat21, rotmat22]
        ])

    @staticmethod
    def rotmat_to_euler(R):
        """Convert rotation matrix to Euler angles"""
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arcsin(-R[2, 0])
        yaw = np.arctan2(R[1, 0], R[0, 0])
        return np.array([roll, pitch, yaw])

    @staticmethod
    def vee_map_3x3(mat):
        """Convert skew-symmetric matrix to vector (vee map)"""
        return np.array([mat[2, 1], mat[0, 2], mat[1, 0]])

    @staticmethod
    def hat_map_3x3(vec):
        """Convert vector to skew-symmetric matrix (hat map)"""
        return np.array([
            [0.0,     -vec[2],  vec[1]],
            [vec[2],   0.0,    -vec[0]],
            [-vec[1],  vec[0],  0.0]
        ])

    @staticmethod
    def rotmat_orthonormalize(R):
        """
        Fast rotation-matrix orthonormalization.
        Ref: "Direction Cosine Matrix IMU: Theory"
        """
        x = R[0, :]
        y = R[1, :]
        error = np.dot(x, y)

        x_orthogonal = x - 0.5 * error * y
        y_orthogonal = y - 0.5 * error * x
        z_orthogonal = np.cross(x_orthogonal, y_orthogonal)

        x_normalized = 0.5 * \
            (3 - np.dot(x_orthogonal, x_orthogonal)) * x_orthogonal
        y_normalized = 0.5 * \
            (3 - np.dot(y_orthogonal, y_orthogonal)) * y_orthogonal
        z_normalized = 0.5 * \
            (3 - np.dot(z_orthogonal, z_orthogonal)) * z_orthogonal

        return np.vstack([x_normalized, y_normalized, z_normalized])

    @staticmethod
    def get_prv_angle(R):
        """Get principal rotation vector angle from a rotation matrix"""
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        return np.arccos(0.5 * (trace - 1.0))
