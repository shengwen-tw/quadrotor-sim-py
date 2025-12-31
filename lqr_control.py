import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import care_sda

from se3_math import NumpySE3


class LQRController:
    def __init__(self, args):
        self.iterations = args.iterations
        self.dt = args.dt

        # LQR weights (match the MATLAB values)
        Q = np.zeros((10, 10))
        Q[0, 0] = 5000.0    # yaw
        Q[1, 1] = 10.0      # roll rate
        Q[2, 2] = 10.0      # pitch rate
        Q[3, 3] = 100.0     # yaw rate
        Q[4, 4] = 2000.0    # vx
        Q[5, 5] = 2000.0    # vy
        Q[6, 6] = 3000.0    # vz
        Q[7, 7] = 10000.0   # x
        Q[8, 8] = 10000.0   # y
        Q[9, 9] = 10000.0   # z
        self.Q = Q

        R = np.zeros((4, 4))
        R[0, 0] = 1.0  # thrust
        R[1, 1] = 1.0  # Mx
        R[2, 2] = 1.0  # My
        R[3, 3] = 1.0  # Mz
        self.R = R

        # Build constant B (12x4) as in MATLAB
        self.B = None  # will build on first run() when inertia/mass known

        # Output selection matrix C (mask-out roll/pitch angles)
        # C maps 12-state -> 10 outputs in the Q space (yaw; p,q,r; u,v,w; x,y,z)
        C = np.zeros((10, 12))
        C[0,  2] = 1.0      # yaw
        C[1,  3] = 1.0      # p
        C[2,  4] = 1.0      # q
        C[3,  5] = 1.0      # r
        C[4,  6] = 1.0      # u
        C[5,  7] = 1.0      # v
        C[6,  8] = 1.0      # w
        C[7,  9] = 1.0      # x
        C[8, 10] = 1.0      # y
        C[9, 11] = 1.0      # z
        self.C = C

        # Reset data buffers
        self.reset()

    def reset(self):
        self.idx = 0
        n = self.iterations
        self.time_arr = np.zeros(n)
        self.sda_time_arr = np.zeros(n)
        self.care_residual_arr = np.zeros(n)
        self.f_arr = np.zeros(n)
        self.M_arr = np.zeros((3, n))
        self.euler_arr = np.zeros((3, n))
        self.W_arr = np.zeros((3, n))
        self.ex_arr = np.zeros((3, n))
        self.ev_arr = np.zeros((3, n))

    # ------- helper: dynamic A matrix identical to MATLAB structure -------
    @staticmethod
    def _A_from_state(J, g, eulers, W, v_b, R):
        """
        Recreate MATLAB's A using the same small-angle-free expressions.
        eulers: (roll, pitch, yaw)
        W: (p, q, r) body rates
        v_b: (u, v, w) body-frame linear velocity
        R: DCM world<-body
        """
        p, q, r = W
        u, v, w = v_b
        phi, theta, psi = eulers

        Ix, Iy, Iz = J[0, 0], J[1, 1], J[2, 2]

        s = np.sin
        c = np.cos
        t = np.tan
        def sec(x): return 1.0 / np.cos(x)

        s_phi, c_phi = s(phi), c(phi)
        s_theta, c_theta = s(theta), c(theta)
        s_psi, c_psi = s(psi), c(psi)
        t_theta = t(theta)
        sec_theta = sec(theta)

        a1 = [-r*s_phi*t_theta + q*c_phi*t_theta,
              r*(c_phi*sec_theta**2) + q*(s_phi*sec_theta**2),
              0, 1, s_phi*t_theta, c_phi*t_theta, 0, 0, 0, 0, 0, 0]
        a2 = [(-q*s_phi - r*c_phi), 0, 0, 0, c_phi, -s_phi, 0, 0, 0, 0, 0, 0]
        a3 = [-r*s_phi/c_theta + q*c_phi/c_theta,
              r*c_phi*sec_theta*t_theta + q*s_phi*sec_theta*t_theta,
              0, 0, s_phi/c_theta, c_phi/c_theta, 0, 0, 0, 0, 0, 0]
        a4 = [0, 0, 0, 0, (Iy-Iz)/Ix * r, (Iy-Iz)/Ix * q, 0, 0, 0, 0, 0, 0]
        a5 = [0, 0, 0, (Iz-Ix)/Iy * r, 0, (Iz-Ix)/Iy * p, 0, 0, 0, 0, 0, 0]
        a6 = [0, 0, 0, (Ix-Iy)/Iz * q, (Ix-Iy)/Iz * p, 0, 0, 0, 0, 0, 0, 0]
        a7 = [0, -g*c_theta, 0, 0, -w, v, 0, r, -q, 0, 0, 0]
        a8 = [g*c_phi*c_theta, -g*s_phi*s_theta,
              0, w, 0, -u, -r, 0, p, 0, 0, 0]
        a9 = [-g*c_theta*s_phi, -g*s_theta *
              c_phi, 0, -v, u, 0, q, -p, 0, 0, 0, 0]
        a10 = [
            w*(c_phi*s_psi - s_phi*c_psi*s_theta) +
            v*(s_phi*s_psi + c_psi*c_phi*s_theta),
            w*(c_phi*c_psi*c_theta) + v *
            (c_psi*s_phi*c_theta) - u*(c_psi*s_theta),
            w*(s_phi*c_psi - c_phi*s_psi*s_theta) - v *
            (c_phi*c_psi - c_phi*c_psi*s_theta) + u*(c_theta*c_psi),
            0, 0, 0,
            c_psi*c_theta,
            (-c_phi*s_psi + c_psi*s_phi*s_theta),
            (s_phi*s_psi + c_phi*c_psi*s_theta),
            0, 0, 0
        ]
        a11 = [
            v*(-s_phi*c_psi + c_phi*s_psi*s_theta) -
            w*(c_psi*c_phi + s_phi*s_psi*s_theta),
            v*(s_phi*s_psi*c_theta) + w *
            (c_phi*s_psi*c_theta) - u*(s_theta*s_psi),
            v*(-c_phi*s_psi + s_phi*c_psi*s_theta) + w *
            (s_psi*s_phi + c_phi*c_psi*s_theta) + u*(c_theta*c_psi),
            0, 0, 0,
            c_theta*s_psi,
            (c_phi*c_psi + s_phi*s_psi*s_theta),
            (-c_psi*s_phi + c_phi*s_psi*s_theta),
            0, 0, 0
        ]
        a12 = [
            -w*s_phi*c_theta + v*c_theta*c_phi,
            -w*c_phi*s_theta - u*c_theta - v*s_theta*s_phi,
            0, 0, 0, 0,
            -s_theta, c_theta*s_phi, c_phi*c_theta,
            0, 0, 0
        ]
        A = np.array([a1, a2, a3, a4, a5, a6, a7, a8,
                     a9, a10, a11, a12], dtype=float)
        return A

    def _ensure_B(self, m, J):
        if self.B is not None:
            return self.B
        Ix, Iy, Iz = J[0, 0], J[1, 1], J[2, 2]
        B = np.zeros((12, 4))
        B[3, 1] = 1.0 / Ix
        B[4, 2] = 1.0 / Iy
        B[5, 3] = 1.0 / Iz
        # thrust affects w-dot (NED/body sign consistent with MATLAB)
        B[8, 0] = -1.0 / m
        self.B = B
        return B

    def _solve_care(self, A, B, H, R):
        t0 = time.perf_counter()
        X = care_sda.care_sda(A, B, H, R)
        t1 = time.perf_counter()
        return X, (t1 - t0)

    def run(self, env):
        # Desired signals (planner must return: xd, vd, ad, yaw_d, Wd, W_dot_d)
        xd, vd, ad, yaw_d, Wd, W_dot_d = env.get_desired_state()

        # === Current states ===
        m = env.uav_dynamics.get_mass()
        J = env.uav_dynamics.get_inertia_matrix()
        g = env.uav_dynamics.get_gravitational_acceleration()
        R = env.uav_dynamics.get_rotmat()
        Rt = R.T
        v = env.uav_dynamics.get_velocity()           # world velocity
        W = env.uav_dynamics.get_angular_velocity()

        # ✅ Correct body-frame velocity
        v_b = Rt @ v

        # Euler angles (roll, pitch, yaw) — must be ZYX
        eulers = NumpySE3.rotmat_to_euler(R)

        # === Linearization ===
        A = self._A_from_state(J, g, eulers, W, v_b, R)
        B = self._ensure_B(m, J)
        C = self.C

        # ✅ Regularize H and R to ensure positive definiteness
        eps = 1e-8
        H = C.T @ self.Q @ C
        H = 0.5 * (H + H.T) + eps * np.eye(H.shape[0])
        R_reg = 0.5 * (self.R + self.R.T) + eps * np.eye(self.R.shape[0])

        # Solve CARE
        X, sda_time = self._solve_care(A, B, H, R_reg)
        G = B @ np.linalg.inv(R_reg) @ B.T

        # CARE residual norm (for diagnostics)
        care_residual = np.linalg.norm(A.T @ X + X @ A - X @ G @ X + H)
        K = np.linalg.inv(R_reg) @ B.T @ X  # 4x12 gain matrix

        # === Compose state vector (12x1) ===
        x = np.zeros(12)
        x[0:3] = eulers
        x[3:6] = W
        x[6:9] = v_b
        x[9:12] = env.uav_dynamics.get_position()

        # === Desired state x0 (12x1) ===
        vd_b = Rt @ vd    # ✅ convert desired velocity to body frame
        x0 = np.zeros(12)
        x0[2] = yaw_d
        x0[3:6] = 0.0
        x0[6:9] = vd_b
        x0[9:12] = xd

        # === Feedforward + Feedback ===
        gravity_ff = np.dot(
            m * g * np.array([0.0, 0.0, 1.0]), R @ np.array([0.0, 0.0, 1.0]))
        u_ff = np.array([gravity_ff, 0.0, 0.0, 0.0])
        error = x - x0
        error[2] = ((eulers[2] - yaw_d + np.pi) % (2 * np.pi)) - np.pi
        u_fb = -K @ error
        u = u_ff + u_fb

        f_total = u[0]
        M_body = u[1:4]

        # ✅ Convert thrust to world-frame force vector
        uav_ctrl_f = f_total * (R @ np.array([0.0, 0.0, 1.0]))
        uav_ctrl_M = M_body

        # === Apply control to dynamics ===
        env.uav_dynamics.set_force(uav_ctrl_f)
        env.uav_dynamics.set_moment(uav_ctrl_M)

        # === Log data ===
        self.time_arr[self.idx] = self.idx * self.dt
        self.sda_time_arr[self.idx] = sda_time
        self.care_residual_arr[self.idx] = care_residual
        self.f_arr[self.idx] = f_total
        self.M_arr[:, self.idx] = uav_ctrl_M
        self.euler_arr[:, self.idx] = np.rad2deg(np.array(eulers))
        self.W_arr[:, self.idx] = np.rad2deg(W)
        self.ex_arr[:, self.idx] = env.uav_dynamics.get_position() - xd
        self.ev_arr[:, self.idx] = env.uav_dynamics.get_velocity() - vd

        self.idx += 1
        return uav_ctrl_M, uav_ctrl_f

    # -------- plotting (沿用/擴充你現有圖表) ----------

    def plot_graph(self):
        t = self.time_arr

        # Control inputs
        plt.figure("Control inputs")
        plt.subplot(4, 1, 1)
        plt.plot(t, self.M_arr[0, :])
        plt.grid(True)
        plt.title("Control inputs")
        plt.ylabel("M_x")
        plt.subplot(4, 1, 2)
        plt.plot(t, self.M_arr[1, :])
        plt.grid(True)
        plt.ylabel("M_y")
        plt.subplot(4, 1, 3)
        plt.plot(t, self.M_arr[2, :])
        plt.grid(True)
        plt.ylabel("M_z")
        plt.subplot(4, 1, 4)
        plt.plot(t, self.f_arr[:])
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("f")

        # Euler angles
        plt.figure("Attitude (Euler)")
        plt.subplot(3, 1, 1)
        plt.plot(t, self.euler_arr[0, :])
        plt.grid(True)
        plt.title("Euler angles")
        plt.ylabel("roll [deg]")
        plt.subplot(3, 1, 2)
        plt.plot(t, self.euler_arr[1, :])
        plt.grid(True)
        plt.ylabel("pitch [deg]")
        plt.subplot(3, 1, 3)
        plt.plot(t, self.euler_arr[2, :])
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("yaw [deg]")

        # Angular rates
        plt.figure("Angular rates (body)")
        plt.subplot(3, 1, 1)
        plt.plot(t, self.W_arr[0, :])
        plt.grid(True)
        plt.title("Angular rate")
        plt.ylabel("p [deg/s]")
        plt.subplot(3, 1, 2)
        plt.plot(t, self.W_arr[1, :])
        plt.grid(True)
        plt.ylabel("q [deg/s]")
        plt.subplot(3, 1, 3)
        plt.plot(t, self.W_arr[2, :])
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("r [deg/s]")

        # Position/Velocity errors
        plt.figure("Position error")
        plt.subplot(3, 1, 1)
        plt.plot(t, self.ex_arr[0, :])
        plt.grid(True)
        plt.title("Position error")
        plt.ylabel("x [m]")
        plt.subplot(3, 1, 2)
        plt.plot(t, self.ex_arr[1, :])
        plt.grid(True)
        plt.ylabel("y [m]")
        plt.subplot(3, 1, 3)
        plt.plot(t, self.ex_arr[2, :])
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("z [m]")

        plt.figure("Velocity error")
        plt.subplot(3, 1, 1)
        plt.plot(t, self.ev_arr[0, :])
        plt.grid(True)
        plt.title("Velocity error")
        plt.ylabel("x [m/s]")
        plt.subplot(3, 1, 2)
        plt.plot(t, self.ev_arr[1, :])
        plt.grid(True)
        plt.ylabel("y [m/s]")
        plt.subplot(3, 1, 3)
        plt.plot(t, self.ev_arr[2, :])
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("z [m/s]")

        # CARE solver diagnostics
        plt.figure("CARE diagnostics")
        plt.subplot(2, 1, 1)
        plt.plot(t, self.sda_time_arr)
        plt.grid(True)
        plt.title("SDA solve time per step")
        plt.ylabel("time [s]")
        plt.subplot(2, 1, 2)
        plt.plot(t, self.care_residual_arr)
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("||CARE residual||")
