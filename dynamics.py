import numpy as np
#from scipy.linalg import expm

from se3_math import SE3


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
        self.R_det = 1.0              # Determinant of R
        self.prv_angle = 0.0          # Principal rotation vector angle

        self.f = np.zeros(3)          # Control force [N]
        self.M = np.zeros(3)          # Control moment [Nm]

        self.math = SE3()

    def set_position(self, x):
        self.x = x

    def set_velocity(self, v):
        self.v = v

    def set_rotmat(self, R):
        self.R = R

    def set_force(self, f):
        self.f = f

    def set_moment(self, M):
        self.M = M

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
        # dR = expm(self.math.hat_map_3x3(self.W * self.dt))
        dR = self.math.hat_map_3x3(self.W * self.dt) + np.eye(3)
        self.R = self.R @ dR
        self.R = self.math.rotmat_orthonormalize(self.R)
        self.R_det = np.linalg.det(self.R)

        # 7. Compute angle of the principal rotation vector (PRV)
        self.prv_angle = self.math.get_prv_angle(self.R)
