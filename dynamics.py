import numpy as np
import torch

from torch import Tensor


class DynamicsBatch:
    def __init__(self, device: str, dt: float, mass: float, J: Tensor, batch: int = 1):
        # Tensor
        self.B = batch
        self.device = torch.device(device)
        self.dtype = torch.float32
        ref = torch.empty((), device=self.device, dtype=self.dtype)

        # Identity matrix
        _I = torch.eye(3, device=self.device, dtype=self.dtype)  # (3, 3)
        self.I = _I.view(1, 3, 3).expand(self.B, 3, 3)  # (B, 3, 3)

        # Z-vector of the inertial frame
        self.e3 = ref.new_zeros((self.B, 3))
        self.e3[:, 2] = 1.0

        self.dt = dt                             # Time step size [s]
        self.mass = mass                         # Mass [kg]
        self.g = 9.8                             # Gravity [m/s^2]
        self.J = J.clone().contiguous()          # Inertia matrix
        self.J_inv = torch.linalg.inv(self.J)    # Inverse inertia matrix
        self.x = ref.new_zeros((self.B, 3))      # Position [m]
        self.v = ref.new_zeros((self.B, 3))      # Velocity [m/s]
        self.a = ref.new_zeros((self.B, 3))      # Acceleration [m/s^2]
        self.R = self.I.clone().contiguous()     # Rotation matrix
        self.W = ref.new_zeros((self.B, 3))      # Angular vel. [rad/s]
        self.W_dot = ref.new_zeros((self.B, 3))  # Angular accel. [rad/s^2]
        self.f = ref.new_zeros((self.B, 3))      # Control force [N]
        self.M = ref.new_zeros((self.B, 3))      # Control moment [N*m]

    def set_position(self, x: Tensor):
        self.x = x.clone().contiguous()

    def set_velocity(self, v: Tensor):
        self.v = v.clone().contiguous()

    def set_force(self, f: Tensor):
        self.f = f.clone().contiguous()

    def set_moment(self, M: Tensor):
        self.M = M.clone().contiguous()

    def set_rotmat(self, R: Tensor):
        self.R = R.clone().contiguous()

    def dv_dt(self, f: Tensor) -> Tensor:
        return (self.mass * self.g * self.e3 - f) / self.mass

    def dW_dt(self, W: Tensor) -> Tensor:
        # JW = (B, 3, 3) @ (B, 3, 1) -> (B, 3, 1)
        JW = (self.J @ W[:, :, None])[:, :, 0]
        # WJW = (B, 3, 1) X (B, 3, 1)
        W_cross_JW = torch.cross(W, JW, dim=1)
        # Wdot = (B, 3, 3) @ ((B, 3, 1) - (B, 3, 1))
        Wdot = (self.J_inv @ (self.M - W_cross_JW)[:, :, None])[:, :, 0]
        return Wdot

    def integrator_euler(self, f_now: Tensor, f_dot: Tensor) -> Tensor:
        """Euler integration"""
        return f_now + f_dot * self.dt

    def integrator_rk4(self, f_now: Tensor, f_dot_func) -> Tensor:
        """Runge–Kutta fourth-order method"""
        k1 = f_dot_func(f_now)
        k2 = f_dot_func(f_now + 0.5 * self.dt * k1)
        k3 = f_dot_func(f_now + 0.5 * self.dt * k2)
        k4 = f_dot_func(f_now + self.dt * k3)
        return f_now + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def hat_map_3x3(self, vec: Tensor) -> Tensor:
        """
        Stack (B, 9) then convert to (B, 3, 3).
        This creates the skew-symmetric matrix (hat map) for cross product
        """
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

    def rotmat_orthonormalize(self, R: Tensor) -> Tensor:
        """
        Fast rotation-matrix orthonormalization (B, 3, 3).
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

    def update(self):
        """
        Update the quadrotor's state for one time step.

        Integration strategy:
        - Linear velocity (v): integrated using Euler method; since
          acceleration is constant within a time step, RK4 would effectively
          reduce to Euler.
        - Position (x): integrated using Euler method; since velocity (v) is
          already updated and treated as constant over the time step, RK4 would
          again reduce to Euler.
        - Angular velocity (W): integrated using RK4 due to nonlinear dynamics
          involving cross products.
        - Rotation matrix (R): updated using the exponential map, which
          preserves the orthogonality and structure of the SO(3) group.
        """

        # 1. Update linear acceleration from force
        self.a = self.dv_dt(self.f)

        # 2. Update angular acceleration from moment
        self.W_dot = self.dW_dt(self.W)

        # 3. Update linear velocity
        self.v = self.integrator_euler(self.v, self.a)

        # 4. Update position
        self.x = self.integrator_euler(self.x, self.v)

        # 5. Update angular velocity
        self.W = self.integrator_rk4(self.W, self.dW_dt)

        # 6. Update rotation matrix
        #dR = torch.matrix_exp(self.hat_map_3x3(self.W * self.dt))
        dR = self.hat_map_3x3(self.W * self.dt) + self.I
        self.R = self.R @ dR
        self.R = self.rotmat_orthonormalize(self.R)


class Dynamics:
    def __init__(self, dt, mass, J=np.eye(3)):
        self._dyn = DynamicsBatch(
            device='cpu',
            dt=dt,
            mass=mass,
            J=torch.as_tensor(J, dtype=torch.float32),
            batch=1
        )
        self.dt = dt
        self.mass = mass
        self.g = 9.8
        self.J = J.copy()
        self.x = np.zeros(3)
        self.v = np.zeros(3)
        self.a = np.zeros(3)
        self.W = np.zeros(3)
        self.W_dot = np.zeros(3)
        self.R = np.eye(3)
        self.f = np.zeros(3)
        self.M = np.zeros(3)
        self._update_internal()

    def _update_internal(self):
        self._dyn.set_position(torch.from_numpy(
            self.x).to(self._dyn.device, self._dyn.dtype))
        self._dyn.set_velocity(torch.from_numpy(
            self.v).to(self._dyn.device, self._dyn.dtype))
        self._dyn.set_rotmat(torch.from_numpy(self.R).to(
            self._dyn.device, self._dyn.dtype))
        self._dyn.set_force(torch.from_numpy(self.f).to(
            self._dyn.device, self._dyn.dtype))
        self._dyn.set_moment(torch.from_numpy(self.M).to(
            self._dyn.device, self._dyn.dtype))

    def state_randomize(self, np_random=None):
        POS_INC_MAX = 1.5
        VEL_INC_MAX = 1.5
        ANG_VEL_INC_MAX = 90  # [deg/s]

        rng = np_random if np_random is not None else np.random

        self.x += rng.uniform(-POS_INC_MAX, POS_INC_MAX, size=3)
        # self.v += rng.uniform(-VEL_INC_MAX, VEL_INC_MAX, size=3)
        # self.W += np.deg2rad(rng.uniform(-ANG_VEL_INC_MAX, ANG_VEL_INC_MAX, size=3))
        # roll = np.deg2rad(rng.uniform(-70, 70))
        # pitch = np.deg2rad(rng.uniform(-70, 70))
        # yaw = np.deg2rad(rng.uniform(-180, 180))
        # self.R = SE3.euler_to_rotmat(roll, pitch, yaw)

        self._update_internal()

    def set_position(self, x):
        self.x = x.copy()
        self._dyn.set_position(torch.from_numpy(
            self.x).to(self._dyn.device, self._dyn.dtype))

    def set_velocity(self, v):
        self.v = v.copy()
        self._dyn.set_velocity(torch.from_numpy(
            self.v).to(self._dyn.device, self._dyn.dtype))

    def set_rotmat(self, R):
        self.R = R.copy()
        self._dyn.set_rotmat(torch.from_numpy(self.R).to(
            self._dyn.device, self._dyn.dtype))

    def set_force(self, f):
        self.f = f.copy()
        self._dyn.set_force(torch.from_numpy(self.f).to(
            self._dyn.device, self._dyn.dtype))

    def set_moment(self, M):
        self.M = M.copy()
        self._dyn.set_moment(torch.from_numpy(self.M).to(
            self._dyn.device, self._dyn.dtype))

    def update(self):
        self._dyn.update()
        self.x = self._dyn.x[0].cpu().numpy().copy()
        self.v = self._dyn.v[0].cpu().numpy().copy()
        self.a = self._dyn.a[0].cpu().numpy().copy()
        self.R = self._dyn.R[0].cpu().numpy().copy()
        self.W = self._dyn.W[0].cpu().numpy().copy()
        self.W_dot = self._dyn.W_dot[0].cpu().numpy().copy()
        self.f = self._dyn.f[0].cpu().numpy().copy()
        self.M = self._dyn.M[0].cpu().numpy().copy()
