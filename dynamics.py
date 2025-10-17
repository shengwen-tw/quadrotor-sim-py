from __future__ import annotations
from typing import Optional, Union
import numpy as np
import torch

Tensor = torch.Tensor
ArrayLike = Union[np.ndarray, Tensor]


def hat_map_3x3(v: Tensor) -> Tensor:
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    O = torch.zeros_like(vx)
    return torch.stack([
        torch.stack([O, -vz, vy], dim=-1),
        torch.stack([vz, O, -vx], dim=-1),
        torch.stack([-vy, vx, O], dim=-1),
    ], dim=-2)


def rotmat_orthonormalize(R: Tensor) -> Tensor:
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
    return torch.stack([x_n, y_n, z_n], dim=-2)


def get_prv_angle(R: Tensor) -> Tensor:
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    c = torch.clamp(0.5 * (tr - 1.0), -1.0, 1.0)
    return torch.arccos(c)


class DynamicsBatch:
    def __init__(self, dt: float, mass: float, J: ArrayLike = None, batch: int = 1,
                 device: Union[str, torch.device] = "cpu", dtype: torch.dtype = torch.float32):
        self.dt = float(dt)
        self.mass = float(mass)
        self.g = 9.8
        self.device = torch.device(device)
        self.dtype = dtype

        if J is None:
            Jm = torch.eye(3, device=self.device, dtype=self.dtype)
        else:
            Jm = torch.as_tensor(J, device=self.device, dtype=self.dtype)
        if Jm.ndim == 2:
            Jm = Jm.expand(batch, 3, 3)
        self.J = Jm.clone()

        B = batch
        zeros3 = torch.zeros(B, 3, device=self.device, dtype=self.dtype)
        self.x = zeros3.clone()
        self.v = zeros3.clone()
        self.a = zeros3.clone()
        self.W = zeros3.clone()
        self.W_dot = zeros3.clone()

        I = torch.eye(3, device=self.device, dtype=self.dtype)
        self.R = I.expand(B, 3, 3).clone()
        self.R_det = torch.ones(B, device=self.device, dtype=self.dtype)
        self.prv_angle = torch.zeros(B, device=self.device, dtype=self.dtype)

        self.f = zeros3.clone()
        self.M = zeros3.clone()

    @property
    def batch(self) -> int:
        return self.x.shape[0]

    def _to_tensor(self, x: ArrayLike, shape_like: Tensor) -> Tensor:
        t = torch.as_tensor(x, device=self.device, dtype=self.dtype)
        if t.ndim == shape_like.ndim - 1:
            t = t.expand(self.batch, *shape_like.shape[1:])
        return t

    def set_position(self, x: ArrayLike):
        self.x = self._to_tensor(x, self.x).clone()

    def set_velocity(self, v: ArrayLike):
        self.v = self._to_tensor(v, self.v).clone()

    def set_force(self, f: ArrayLike):
        self.f = self._to_tensor(f, self.f).clone()

    def set_moment(self, M: ArrayLike):
        self.M = self._to_tensor(M, self.M).clone()

    def set_rotmat(self, R: ArrayLike):
        t = torch.as_tensor(R, device=self.device, dtype=self.dtype)
        if t.ndim == 2:
            t = t.expand(self.batch, 3, 3)
        self.R = rotmat_orthonormalize(t.clone())

    def dv_dt(self, f: Tensor) -> Tensor:
        e3 = torch.tensor([0.0, 0.0, 1.0], device=self.device,
                          dtype=self.dtype).expand(self.batch, 3)
        return (self.mass * self.g * e3 - f) / self.mass

    def dW_dt(self, W: Tensor) -> Tensor:
        JW = torch.einsum('bij,bj->bi', self.J, W)
        W_cross_JW = torch.cross(W, JW, dim=-1)
        J_inv = torch.linalg.inv(self.J)
        return torch.einsum('bij,bj->bi', J_inv, (self.M - W_cross_JW))

    def integrator_euler(self, f_now: Tensor, f_dot: Tensor, dt=None) -> Tensor:
        if dt is None:
            dt = self.dt
        return f_now + f_dot * dt

    def integrator_rk4(self, f_now: Tensor, f_dot_func, dt: float) -> Tensor:
        k1 = f_dot_func(f_now)
        k2 = f_dot_func(f_now + 0.5 * dt * k1)
        k3 = f_dot_func(f_now + 0.5 * dt * k2)
        k4 = f_dot_func(f_now + dt * k3)
        return f_now + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def update(self):
        self.a = self.dv_dt(self.f)
        self.W_dot = self.dW_dt(self.W)
        self.v = self.integrator_euler(self.v, self.a)
        self.x = self.integrator_euler(self.x, self.v)
        self.W = self.integrator_rk4(self.W, self.dW_dt, self.dt)
        dR = torch.matrix_exp(hat_map_3x3(self.W * self.dt))
        self.R = self.R @ dR
        self.R = rotmat_orthonormalize(self.R)
        self.R_det = torch.det(self.R)
        self.prv_angle = get_prv_angle(self.R)


class Dynamics:
    def __init__(self, dt: float, mass: float, J: ArrayLike = None):
        self._dyn = DynamicsBatch(dt, mass, J, batch=1)
        self.dt = self._dyn.dt
        self.mass = self._dyn.mass
        self.g = self._dyn.g
        self.J = self._dyn.J[0].detach().cpu().numpy()
        self.sync_state()

    def sync_state(self):
        self.x = self._dyn.x[0].detach().cpu().numpy()
        self.v = self._dyn.v[0].detach().cpu().numpy()
        self.a = self._dyn.a[0].detach().cpu().numpy()
        self.R = self._dyn.R[0].detach().cpu().numpy()
        self.W = self._dyn.W[0].detach().cpu().numpy()
        self.W_dot = self._dyn.W_dot[0].detach().cpu().numpy()
        self.f = self._dyn.f[0].detach().cpu().numpy()
        self.M = self._dyn.M[0].detach().cpu().numpy()

    def set_position(self, x): self._dyn.set_position(x)
    def set_velocity(self, v): self._dyn.set_velocity(v)
    def set_rotmat(self, R): self._dyn.set_rotmat(R)
    def set_force(self, f): self._dyn.set_force(f)
    def set_moment(self, M): self._dyn.set_moment(M)

    def update(self):
        self._dyn.update()
        self.sync_state()
