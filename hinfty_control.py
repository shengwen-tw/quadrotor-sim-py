import argparse
import time
import matplotlib.pyplot as plt
import numpy as np

from hinf_syn import hinf_syn
from se3_math import NumpySE3


class HinfController:
    def __init__(self, args):
        self.iterations = args.iterations
        self.dt = args.dt

        # Lower bound γ for the H∞ synthesis.
        # Use 0 to let `hinf_syn` estimate the theoretical lower bound, then set a
        # practical value (typically slightly larger) for a numerically robust
        # suboptimal solution.
        self.gamma_lb = 30

        # Disturbance matrix
        self.B1 = None

        # Control matrix
        self.B2 = None

        # Output weighting matrix (C1)
        self.C1 = np.zeros((14, 12))
        self.C1[0, 2] = 125   # Waw
        self.C1[1, 3] = 10    # Roll rate
        self.C1[2, 4] = 10    # Pitch rate
        self.C1[3, 5] = 25    # Yaw rate
        self.C1[4, 6] = 50    # vx (body-fixed frame)
        self.C1[5, 7] = 50    # vy (body-fixed frame)
        self.C1[6, 8] = 100   # vz (body-fixed frame)
        self.C1[7, 9] = 200   # x (inertial frame)
        self.C1[8, 10] = 200  # y (inertial frame)
        self.C1[9, 11] = 160  # z (inertial frame)

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
        self.hinf_time_arr = np.zeros(self.iterations)
        self.ric_residual_arr = np.zeros(self.iterations)
        self.gamma_arr = np.zeros(self.iterations)
        self.gamma_lb_arr = np.zeros(self.iterations)

    def build_disturbance_matrix(self, mass, J):
        B1 = np.zeros((12, 6))
        B1[3, 3] = 1.0 / J[0, 0]
        B1[4, 4] = 1.0 / J[1, 1]
        B1[5, 5] = 1.0 / J[2, 2]
        B1[6, 0] = 1.0 / mass
        B1[7, 1] = 1.0 / mass
        B1[8, 2] = 1.0 / mass
        return B1

    def build_control_matrix(self, mass, J):
        B2 = np.zeros((12, 4))
        B2[3, 1] = 1.0 / J[0, 0]
        B2[4, 2] = 1.0 / J[1, 1]
        B2[5, 3] = 1.0 / J[2, 2]
        B2[8, 0] = -1.0 / mass
        return B2

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
        A[0, 1] = r*(c_phi*(sec_theta**2) + q*s_phi*(sec_theta**2))
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
        A[6, 1] = -g * c_theta
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

        # A[10, :]
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
        if self.B1 is None:
            self.B1 = self.build_disturbance_matrix(mass, J)
        if self.B2 is None:
            self.B2 = self.build_control_matrix(mass, J)

        # H-infinity synthesis
        start_time = time.time()
        gamma, gamma_lb, X, ric_residual = \
            hinf_syn(A, self.B1, self.B2, self.C1, gamma_lb=self.gamma_lb)
        hinf_time = time.time() - start_time

        # Compute optimal feedback gain matrix K
        K = -self.B2.T @ X

        # Construct current state vector
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
        u_fb = K @ error

        # Combine feedforward and feedback control
        u = u_ff + u_fb

        # Control force (3x1 vector in world frame)
        uav_ctrl_f = u[0] * (R @ self.e3)

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
        self.hinf_time_arr[self.idx] = hinf_time
        self.ric_residual_arr[self.idx] = ric_residual
        self.gamma_arr[self.idx] = gamma
        self.gamma_lb_arr[self.idx] = gamma_lb

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

        # Plot H-infinity solver statistics
        plt.figure("H-infinity statistics")
        plt.subplot(4, 1, 1)
        plt.plot(t, self.hinf_time_arr)
        plt.grid(True)
        plt.title("H-infinity synthesis time per step")
        plt.ylabel("time [s]")
        plt.subplot(4, 1, 2)
        plt.plot(t, self.ric_residual_arr)
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("||Riccati residual||")
        plt.subplot(4, 1, 3)
        plt.plot(t, self.gamma_arr)
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("gamma")
        plt.subplot(4, 1, 4)
        plt.plot(t, self.gamma_lb_arr)
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("gamma_lb")
