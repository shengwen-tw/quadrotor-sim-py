import argparse
import time
import matplotlib.pyplot as plt
import numpy as np

from care_sda import care_sda
from se3_math import NumpySE3


class LQRController:
    def __init__(self, args):
        self.iterations = args.iterations
        self.dt = args.dt

        # State error penalty weights
        Q = np.zeros((10, 10))
        Q[0, 0] = 5000.0   # Yaw
        Q[1, 1] = 10.0     # Roll rate
        Q[2, 2] = 10.0     # Pitch rate
        Q[3, 3] = 100.0    # Yaw rate
        Q[4, 4] = 2000.0   # vx (body-fixed frame)
        Q[5, 5] = 2000.0   # vy (body-fixed frame)
        Q[6, 6] = 3000.0   # vz (body-fixed frame)
        Q[7, 7] = 10000.0  # x (inertial frame)
        Q[8, 8] = 10000.0  # y (inertial frame)
        Q[9, 9] = 10000.0  # z (inertial frame)
        self.Q = Q

        # Control weights
        R = np.zeros((4, 4))
        R[0, 0] = 1.0  # Thrust
        R[1, 1] = 1.0  # Mx
        R[2, 2] = 1.0  # My
        R[3, 3] = 1.0  # Mz
        self.R = R

        # Control matrix
        self.B = None  # Build later when mass and inertia matrix is known

        # Observation matrix
        C = np.zeros((10, 12))
        C[0,  2] = 1.0  # Yaw
        C[1,  3] = 1.0  # Roll rate
        C[2,  4] = 1.0  # Pitch rate
        C[3,  5] = 1.0  # Yaw rate
        C[4,  6] = 1.0  # vx (body-fixed frame)
        C[5,  7] = 1.0  # vy (body-fixed frame)
        C[6,  8] = 1.0  # vz (body-fixed frame)
        C[7,  9] = 1.0  # x (inertial frame)
        C[8, 10] = 1.0  # y (inertial frame)
        C[9, 11] = 1.0  # z (inertial frame)
        self.C = C

        # Z-vector of the inertial frame
        self.e3 = np.array([0.0, 0.0, 1.0])

        # Reset data
        self.reset()

    def reset(self):
        # Reset time index
        self.idx = 0

        # Data for plotting
        self.time_arr = np.zeros(self.iterations)
        self.euler_arr = np.zeros((3, self.iterations))
        self.W_arr = np.zeros((3, self.iterations))
        self.ex_arr = np.zeros((3, self.iterations))
        self.ev_arr = np.zeros((3, self.iterations))
        self.M_arr = np.zeros((3, self.iterations))
        self.f_arr = np.zeros(self.iterations)
        self.sda_time_arr = np.zeros(self.iterations)
        self.care_residual_arr = np.zeros(self.iterations)

    def compute_state_transition_matrix(self, J, g, eulers, W, v_b):
        p = W[0]
        q = W[1]
        r = W[2]
        u = v_b[0]
        v = v_b[1]
        w = v_b[2]
        phi = eulers[0]
        theta = eulers[1]
        psi = eulers[2]
        Jx = J[0, 0]
        Jy = J[1, 1]
        Jz = J[2, 2]

        s_phi = np.sin(phi)
        c_phi = np.cos(phi)
        s_theta = np.sin(theta)
        c_theta = np.cos(theta)
        s_psi = np.sin(psi)
        c_psi = np.cos(psi)
        t_theta = np.tan(theta)
        sec_theta = 1.0 / np.cos(theta)

        A = np.zeros((12, 12), dtype=float)

        # A[0, :]
        A[0, 0] = -r*s_phi*t_theta + q*c_phi*t_theta
        A[0, 1] = r*(c_phi*sec_theta**2) + q*(s_phi*sec_theta**2)
        A[0, 3] = 1
        A[0, 4] = s_phi*t_theta
        A[0, 5] = c_phi*t_theta

        # A[1, :]
        A[1, 0] = -q*s_phi - r*c_phi
        A[1, 4] = c_phi
        A[1, 5] = -s_phi

        # A[2, :]
        A[2, 0] = -r*s_phi/c_theta + q*c_phi/c_theta
        A[2, 1] = r*c_phi*sec_theta*t_theta + q*s_phi*sec_theta*t_theta
        A[2, 4] = s_phi/c_theta
        A[2, 5] = c_phi/c_theta

        # A[3, :]
        A[3, 4] = (Jy - Jz) / Jx * r
        A[3, 5] = (Jy - Jz) / Jx * q

        # A[4, :]
        A[4, 3] = (Jz - Jx) / Jy * r
        A[4, 5] = (Jz - Jx) / Jy * p

        # A[5, :]
        A[5, 3] = (Jx - Jy) / Jz * q
        A[5, 4] = (Jx - Jy) / Jz * p

        # A[6, :]
        A[6, 1] = -g*c_theta
        A[6, 4] = -w
        A[6, 5] = v
        A[6, 7] = r
        A[6, 8] = -q

        # A[7, :]
        A[7, 0] = g*c_phi*c_theta
        A[7, 1] = -g*s_phi*s_theta
        A[7, 3] = w
        A[7, 5] = -u
        A[7, 6] = -r
        A[7, 8] = p

        # A[8, :]
        A[8, 0] = -g*c_theta*s_phi
        A[8, 1] = -g*s_theta*c_phi
        A[8, 3] = -v
        A[8, 4] = u
        A[8, 6] = q
        A[8, 7] = -p

        # A[9, :]
        A[9, 0] = (
            w*(c_phi*s_psi - s_phi*c_psi*s_theta) +
            v*(s_phi*s_psi + c_psi*c_phi*s_theta)
        )
        A[9, 1] = (
            w*(c_phi*c_psi*c_theta) +
            v*(c_psi*s_phi*c_theta) -
            u*(c_psi*s_theta)
        )
        A[9, 2] = (
            w*(s_phi*c_psi - c_phi*s_psi*s_theta) -
            v*(c_phi*c_psi - c_phi*c_psi*s_theta) +
            u*(c_theta*c_psi)
        )
        A[9, 6] = c_psi*c_theta
        A[9, 7] = -c_phi*s_psi + c_psi*s_phi*s_theta
        A[9, 8] = s_phi*s_psi + c_phi*c_psi*s_theta

        # A[10,:]
        A[10, 0] = (
            v*(-s_phi*c_psi + c_phi*s_psi*s_theta) -
            w*(c_psi*c_phi + s_phi*s_psi*s_theta)
        )
        A[10, 1] = (
            v*(s_phi*s_psi*c_theta) +
            w*(c_phi*s_psi*c_theta) -
            u*(s_theta*s_psi)
        )
        A[10, 2] = (
            v*(-c_phi*s_psi + s_phi*c_psi*s_theta) +
            w*(s_psi*s_phi + c_phi*c_psi*s_theta) +
            u*(c_theta*c_psi)
        )
        A[10, 6] = c_theta*s_psi
        A[10, 7] = c_phi*c_psi + s_phi*s_psi*s_theta
        A[10, 8] = -c_psi*s_phi + c_phi*s_psi*s_theta

        # A[11, :]
        A[11, 0] = -w*s_phi*c_theta + v*c_theta*c_phi
        A[11, 1] = -w*c_phi*s_theta - u*c_theta - v*s_theta*s_phi
        A[11, 6] = -s_theta
        A[11, 7] = c_theta*s_phi
        A[11, 8] = c_phi*c_theta

        return A

    def compute_control_matrix(self, m, J):
        if self.B is not None:
            return self.B

        self.B = np.zeros((12, 4))
        self.B[3, 1] = 1.0 / J[0, 0]
        self.B[4, 2] = 1.0 / J[1, 1]
        self.B[5, 3] = 1.0 / J[2, 2]
        self.B[8, 0] = -1.0 / m

        return self.B

    def solve_care(self, A, B, H, R):
        t0 = time.perf_counter()
        G = B @ np.linalg.inv(R) @ B.T  # TODO
        X = care_sda(A, H, G)
        t1 = time.perf_counter()
        elapsed_time = t1 - t0
        return X, elapsed_time

    def run(self, env):
        # Desired values (i.e., reference signals)
        [xd, vd, ad, yaw_d, Wd, W_dot_d] = env.get_desired_state()

        # States and parameters
        mass = env.uav_dynamics.get_mass()
        J = env.uav_dynamics.get_inertia_matrix()
        g = env.uav_dynamics.get_gravitational_acceleration()
        R = env.uav_dynamics.get_rotmat()
        Rt = R.T
        x = env.uav_dynamics.get_position()
        v = env.uav_dynamics.get_velocity()  # Inertial frame
        v_b = Rt @ v                         # body-fixed frame
        W = env.uav_dynamics.get_angular_velocity()
        eulers = NumpySE3.rotmat_to_euler(R)

        # System linearization
        A = self.compute_state_transition_matrix(J, g, eulers, W, v_b)
        B = self.compute_control_matrix(mass, J)
        C = self.C

        # Solve CARE for optimal feedback gain matrix K
        H = C.T @ self.Q @ C
        X, sda_time = self.solve_care(A, B, H, self.R)
        K = np.linalg.inv(self.R) @ B.T @ X

        # Compute CARE residual norm
        G = B @ np.linalg.inv(self.R) @ B.T
        care_residual = np.linalg.norm(A.T @ X + X @ A - X @ G @ X + H)

        # Construct state vector
        x_state = np.zeros(12)
        x_state[0:3] = eulers
        x_state[3:6] = W
        x_state[6:9] = v_b
        x_state[9:12] = x

        # Construct target state vector (i.e., reference signal)
        x0 = np.zeros(12)
        x0[2] = yaw_d      # Desired yaw angle
        x0[3:6] = 0.0      # Desired attitude rate
        x0[6:9] = Rt @ vd  # Desired velocity in the body-fixed frame
        x0[9:12] = xd      # Desired position in the inertial frame

        # Compute feedforward term caused by constant mass and gravity (weight)
        gravity_ff = np.dot(mass * g * self.e3, R @ self.e3)
        u_ff = np.array([gravity_ff, 0.0, 0.0, 0.0])

        # Compute state error
        error = x_state - x0

        # Wrap yaw error between [-pi, pi]
        error[2] = ((error[2] + np.pi) % (2 * np.pi)) - np.pi

        # Compute feedback control
        u_fb = -K @ error

        # Combine feedforward and feedback control
        u = u_ff + u_fb

        # Control force (3x1 vector in world frame)
        uav_ctrl_f = u[0] * R @ self.e3

        # Control moment (3x1 vector in body-fixed frame)
        uav_ctrl_M = u[1:4]

        # Record data for plotting
        self.time_arr[self.idx] = self.idx * self.dt
        self.euler_arr[:, self.idx] = np.rad2deg(np.array(eulers))
        self.W_arr[:, self.idx] = np.rad2deg(W)
        self.ex_arr[:, self.idx] = x - xd
        self.ev_arr[:, self.idx] = v - vd
        self.M_arr[:, self.idx] = uav_ctrl_M
        self.f_arr[self.idx] = u[0]
        self.sda_time_arr[self.idx] = sda_time
        self.care_residual_arr[self.idx] = care_residual

        # Update time index
        self.idx += 1

        return uav_ctrl_M, uav_ctrl_f

    def plot_graph(self):
        t = self.time_arr

        # Plot control inputs
        plt.figure("Control inputs")
        plt.subplot(4, 1, 1)
        plt.plot(t, self.M_arr[0, :])
        plt.grid(True)
        plt.title("Control inputs")
        plt.xlabel("time [s]")
        plt.ylabel("M_x")
        plt.subplot(4, 1, 2)
        plt.plot(t, self.M_arr[1, :])
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("M_y")
        plt.subplot(4, 1, 3)
        plt.plot(t, self.M_arr[2, :])
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("M_z")
        plt.subplot(4, 1, 4)
        plt.plot(t, self.f_arr[:])
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("f")

        # Plot angular rates
        plt.figure("Angular rates (body)")
        plt.subplot(3, 1, 1)
        plt.plot(t, self.W_arr[0, :])
        plt.grid(True)
        plt.title("Angular rate")
        plt.xlabel("time [s]")
        plt.ylabel("p [deg/s]")
        plt.subplot(3, 1, 2)
        plt.plot(t, self.W_arr[1, :])
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("q [deg/s]")
        plt.subplot(3, 1, 3)
        plt.plot(t, self.W_arr[2, :])
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("r [deg/s]")

        # Plot position error
        plt.figure("Position error")
        plt.subplot(3, 1, 1)
        plt.plot(t, self.ex_arr[0, :])
        plt.grid(True)
        plt.title("Position error")
        plt.xlabel("time [s]")
        plt.ylabel("x [m]")
        plt.subplot(3, 1, 2)
        plt.plot(t, self.ex_arr[1, :])
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("y [m]")
        plt.subplot(3, 1, 3)
        plt.plot(t, self.ex_arr[2, :])
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("z [m]")

        # Plot velocity error
        plt.figure("Velocity error")
        plt.subplot(3, 1, 1)
        plt.plot(t, self.ev_arr[0, :])
        plt.grid(True)
        plt.title("Velocity error")
        plt.xlabel("time [s]")
        plt.ylabel("x [m/s]")
        plt.subplot(3, 1, 2)
        plt.plot(t, self.ev_arr[1, :])
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("y [m/s]")
        plt.subplot(3, 1, 3)
        plt.plot(t, self.ev_arr[2, :])
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("z [m/s]")

        # Plot CARE solver statistics
        plt.figure("CARE statistics")
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
