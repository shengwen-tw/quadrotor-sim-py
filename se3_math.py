import numpy as np


class SE3:
    @staticmethod
    def euler_to_rotmat(roll, pitch, yaw):
        """Convert Euler angles (roll, pitch, yaw) to rotation matrix"""
        cos_phi = np.cos(roll)
        cos_theta = np.cos(pitch)
        cos_psi = np.cos(yaw)

        sin_phi = np.sin(roll)
        sin_theta = np.sin(pitch)
        sin_psi = np.sin(yaw)

        rotmat11 = cos_theta * cos_psi
        rotmat12 = -cos_phi * sin_psi + sin_phi * sin_theta * cos_psi
        rotmat13 = sin_phi * sin_psi + cos_phi * sin_theta * cos_psi

        rotmat21 = cos_theta * sin_psi
        rotmat22 = cos_phi * cos_psi + sin_phi * sin_theta * sin_psi
        rotmat23 = -sin_phi * cos_psi + cos_phi * sin_theta * sin_psi

        rotmat31 = -sin_theta
        rotmat32 = sin_phi * cos_theta
        rotmat33 = cos_phi * cos_theta

        return np.array([
            [rotmat11, rotmat12, rotmat13],
            [rotmat21, rotmat22, rotmat23],
            [rotmat31, rotmat32, rotmat33]
        ])

    @staticmethod
    def rotmat_to_euler(R):
        """Convert rotation matrix to Euler angles (roll, pitch, yaw)"""
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arcsin(-R[2, 0])
        yaw = np.arctan2(R[1, 0], R[0, 0])
        return np.array([roll, pitch, yaw])

    @staticmethod
    def vee_map_3x3(mat):
        """Convert skew-symmetric matrix to vector (vee operator)"""
        return np.array([
            mat[2, 1],
            mat[0, 2],
            mat[1, 0]
        ])

    @staticmethod
    def hat_map_3x3(vec):
        """Convert vector to skew-symmetric matrix (hat operator)"""
        return np.array([
            [0.0,     -vec[2],  vec[1]],
            [vec[2],   0.0,    -vec[0]],
            [-vec[1],  vec[0],  0.0]
        ])

    @staticmethod
    def rotmat_orthonormalize(R):
        """
        Orthonormalize a 3x3 rotation matrix using fast approximation.
        Based on: "Direction Cosine Matrix IMU: Theory"
        Ensures the matrix remains in SO(3) with minimal computational cost.
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
        angle_rad = np.arccos(0.5 * (trace - 1))
        return angle_rad
