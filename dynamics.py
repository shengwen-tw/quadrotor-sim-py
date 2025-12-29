from __future__ import annotations

import numpy as np
import torch

from typing import Optional, Union
from scipy.linalg import expm
from se3_math import SE3


Tensor = torch.Tensor


def hat_map_3x3(v: Tensor) -> Tensor:
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    O = torch.zeros_like(vx)
    return torch.stack([
        torch.stack([O, -vz,  vy], dim=-1),
        torch.stack([vz,  O, -vx], dim=-1),
        torch.stack([-vy, vx,  O], dim=-1),
    ], dim=-2)  # (..., 3, 3)


def rotmat_orthonormalize(R: Tensor) -> Tensor:
    # Fast-ish Gram-Schmidt + NR normalize (same as your version)
    x = R[..., 0, :]
    y = R[..., 1, :]
    error = (x * y).sum(dim=-1, keepdim=True)
    x_orth = x - 0.5 * error * y
    y_orth = y - 0.5 * error * x
    z_orth = torch.cross(x_orth, y_orth, dim=-1)

    def _nr_normalize(v: Tensor) -> Tensor:
        v2 = (v * v).sum(dim=-1, keepdim=True)
        return 0.5 * (3.0 - v2) * v

    x_n = _nr_normalize(x_orth)
    y_n = _nr_normalize(y_orth)
    z_n = _nr_normalize(z_orth)
    return torch.stack([x_n, y_n, z_n], dim=-2)  # (..., 3, 3)


class DynamicsBatch:
    """
    Pure-tensor batched dynamics.
    - All states/inputs are torch.Tensor on the same device/dtype.
    - No NumPy, no CPU sync.
    """

    def __init__(
        self,
        dt: float,
        mass: float,
        J: Optional[Tensor] = None,
        batch: int = 1,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available(
        ) else "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.dt = float(dt)
        self.mass = float(mass)
        self.g = 9.8

        self.device = torch.device(device)
        self.dtype = dtype

        B = int(batch)

        # Inertia
        if J is None:
            Jm = torch.eye(3, device=self.device,
                           dtype=self.dtype).expand(B, 3, 3).clone()
        else:
            if not torch.is_tensor(J):
                raise TypeError(
                    "J must be a torch.Tensor (pure tensor version).")
            Jm = J.to(device=self.device, dtype=self.dtype)
            if Jm.ndim == 2:
                Jm = Jm.expand(B, 3, 3)
            elif Jm.ndim != 3 or Jm.shape[-2:] != (3, 3):
                raise ValueError("J must have shape (3,3) or (B,3,3).")
            if Jm.shape[0] != B:
                # allow broadcasting only if J was (3,3); otherwise mismatch is likely a bug
                raise ValueError(f"J batch {Jm.shape[0]} != batch {B}.")
            Jm = Jm.contiguous()

        self.J = Jm
        # precompute (big speedup vs inv every step)
        self.J_inv = torch.linalg.inv(self.J)

        # Constants
        self.e3 = torch.tensor(
            [0.0, 0.0, 1.0], device=self.device, dtype=self.dtype).view(1, 3).expand(B, 3)

        # State
        self.x = torch.zeros(B, 3, device=self.device, dtype=self.dtype)
        self.v = torch.zeros(B, 3, device=self.device, dtype=self.dtype)
        self.a = torch.zeros(B, 3, device=self.device, dtype=self.dtype)

        self.W = torch.zeros(B, 3, device=self.device, dtype=self.dtype)
        self.W_dot = torch.zeros(B, 3, device=self.device, dtype=self.dtype)

        I = torch.eye(3, device=self.device, dtype=self.dtype).view(
            1, 3, 3).expand(B, 3, 3)
        self.R = I.clone()

        # Inputs
        self.f = torch.zeros(B, 3, device=self.device, dtype=self.dtype)
        self.M = torch.zeros(B, 3, device=self.device, dtype=self.dtype)

    @property
    def batch(self) -> int:
        return self.x.shape[0]

    def _check_vec(self, t: Tensor, name: str) -> Tensor:
        if t.ndim == 1:
            if t.shape[0] != 3:
                raise ValueError(f"{name} must be shape (3,) or (B,3).")
            return t.view(1, 3).expand(self.batch, 3)
        if t.ndim == 2:
            if t.shape != (self.batch, 3):
                raise ValueError(
                    f"{name} must be shape (B,3); got {tuple(t.shape)}.")
            return t
        raise ValueError(f"{name} must be a vector tensor; got ndim={t.ndim}.")

    def _check_R(self, R: Tensor) -> Tensor:
        if R.ndim == 2:
            if R.shape != (3, 3):
                raise ValueError("R must be (3,3) or (B,3,3).")
            return R.view(1, 3, 3).expand(self.batch, 3, 3)
        if R.ndim == 3:
            if R.shape != (self.batch, 3, 3):
                raise ValueError(f"R must be (B,3,3); got {tuple(R.shape)}.")
            return R
        raise ValueError("R must be (3,3) or (B,3,3).")

    # --- setters (pure tensor) ---
    def set_position(self, x: Tensor):
        if not torch.is_tensor(x):
            raise TypeError("x must be torch.Tensor.")
        self.x = self._check_vec(
            x.to(self.device, self.dtype), "x").contiguous()

    def set_velocity(self, v: Tensor):
        if not torch.is_tensor(v):
            raise TypeError("v must be torch.Tensor.")
        self.v = self._check_vec(
            v.to(self.device, self.dtype), "v").contiguous()

    def set_force(self, f: Tensor):
        if not torch.is_tensor(f):
            raise TypeError("f must be torch.Tensor.")
        self.f = self._check_vec(
            f.to(self.device, self.dtype), "f").contiguous()

    def set_moment(self, M: Tensor):
        if not torch.is_tensor(M):
            raise TypeError("M must be torch.Tensor.")
        self.M = self._check_vec(
            M.to(self.device, self.dtype), "M").contiguous()

    def set_rotmat(self, R: Tensor, orthonormalize: bool = True):
        if not torch.is_tensor(R):
            raise TypeError("R must be torch.Tensor.")
        Rt = self._check_R(R.to(self.device, self.dtype)).contiguous()
        self.R = rotmat_orthonormalize(Rt) if orthonormalize else Rt

    # --- dynamics ---
    def dv_dt(self, f: Tensor) -> Tensor:
        # a = (m g e3 - f) / m
        return (self.mass * self.g * self.e3 - f) / self.mass

    def dW_dt(self, W: Tensor) -> Tensor:
        # W_dot = J^{-1} ( M - W x (J W) )
        JW = torch.einsum("bij,bj->bi", self.J, W)
        W_cross_JW = torch.cross(W, JW, dim=-1)
        return torch.einsum("bij,bj->bi", self.J_inv, (self.M - W_cross_JW))

    def integrator_euler(self, f_now: Tensor, f_dot: Tensor, dt: Optional[float] = None) -> Tensor:
        h = self.dt if dt is None else float(dt)
        return f_now + f_dot * h

    def integrator_rk4(self, f_now: Tensor, f_dot_func, dt: float) -> Tensor:
        k1 = f_dot_func(f_now)
        k2 = f_dot_func(f_now + 0.5 * dt * k1)
        k3 = f_dot_func(f_now + 0.5 * dt * k2)
        k4 = f_dot_func(f_now + dt * k3)
        return f_now + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def update(self, orthonormalize_R: bool = True):
        # 1) linear accel
        self.a = self.dv_dt(self.f)

        # 2) angular accel (optional bookkeeping)
        self.W_dot = self.dW_dt(self.W)

        # 3) v, x (semi-implicit Euler: use updated v)
        self.v = self.v + self.a * self.dt
        self.x = self.x + self.v * self.dt

        # 4) W (RK4)
        self.W = self.integrator_rk4(self.W, self.dW_dt, self.dt)

        # 5) R (matrix exponential on so(3))
        dR = torch.matrix_exp(hat_map_3x3(self.W * self.dt))
        self.R = self.R @ dR

        if orthonormalize_R:
            self.R = rotmat_orthonormalize(self.R)


class Dynamics:
    def __init__(self, dt, mass, J=np.eye(3)):
        self.dt = dt                  # Time step [sec]
        self.mass = mass              # Mass [kg]
        self.g = 9.8                  # Gravity [m/s^2]
        self.J = J                    # Inertia matrix [kg*m^2]

        self.x = np.zeros(3)          # Position
        self.v = np.zeros(3)          # Velocity
        self.a = np.zeros(3)          # Acceleration

        self.W = np.zeros(3)          # Angular velocity
        self.W_dot = np.zeros(3)      # Angular acceleration
        self.R = np.eye(3)            # Rotation matrix

        self.f = np.zeros(3)          # Control force [N]
        self.M = np.zeros(3)          # Control moment [Nm]

        self.math = SE3()

    def state_randomize(self, np_random=None):
        POS_INC_MAX = 1.5
        VEL_INC_MAX = 1.5
        ANG_VEL_INC_MAX = 90  # [deg/s]

        if np_random == None:
            rng = np.random
        else:
            rng = np_random

        self.x += rng.uniform(-POS_INC_MAX, POS_INC_MAX, size=3)
        #self.v += rng.uniform(-VEL_INC_MAX, VEL_INC_MAX, size=3)
        # self.W += np.deg2rad(rng.uniform(-ANG_VEL_INC_MAX,
        #                     ANG_VEL_INC_MAX, size=3))

        #roll = np.deg2rad(rng.uniform(-70, 70))
        #pitch = np.deg2rad(rng.uniform(-70, 70))
        #yaw = np.deg2rad(rng.uniform(-180, 180))
        #self.R = SE3.euler_to_rotmat(roll, pitch, yaw)

    def set_position(self, x):
        self.x = x.copy()

    def set_velocity(self, v):
        self.v = v.copy()

    def set_rotmat(self, R):
        self.R = R.copy()

    def set_force(self, f):
        self.f = f.copy()

    def set_moment(self, M):
        self.M = M.copy()

    def integrator_euler(self, f_now, f_dot, dt=None):
        """Euler integration"""
        if dt is None:
            dt = self.dt
        return f_now + f_dot * dt

    def integrator_rk4(self, f_now, f_dot_func, dt):
        """Runge–Kutta fourth-order method"""
        k1 = f_dot_func(f_now)
        k2 = f_dot_func(f_now + 0.5 * dt * k1)
        k3 = f_dot_func(f_now + 0.5 * dt * k2)
        k4 = f_dot_func(f_now + dt * k3)
        return f_now + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def dv_dt(self, f):
        e3 = np.array([0, 0, 1])
        a = (self.mass * self.g * e3 - f) / self.mass
        return a

    def dW_dt(self, W):
        JW = self.J @ W
        W_cross_JW = np.cross(W, JW)
        W_dot = np.linalg.inv(self.J) @ (self.M - W_cross_JW)
        return W_dot

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
        self.W = self.integrator_rk4(self.W, self.dW_dt, self.dt)

        # 6. Update rotation matrix with exponential mapping
        dR = expm(self.math.hat_map_3x3(self.W * self.dt))
        #dR = self.math.hat_map_3x3(self.W * self.dt) + np.eye(3)
        self.R = self.R @ dR
        self.R = self.math.rotmat_orthonormalize(self.R)
