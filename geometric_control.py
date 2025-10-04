import argparse
import matplotlib.pyplot as plt
import numpy as np

from dynamics import Dynamics
from rigidbody_visualize import QuadRenderer
from se3_math import SE3
from trajectory_planner import TrajectoryPlanner


class GeometricMomentController:
    """
    A Geometric tracking controller that only implements the moment control part
    """

    def __init__(self):
        # Controller gains
        self.kR = np.array([10.0, 10.0, 10.0])
        self.kW = np.array([2.0, 2.0, 2.0])

    def compute(self, uav_dynamics: Dynamics, target):
        # States and parameters
        mass = uav_dynamics.mass
        J = uav_dynamics.J
        R = uav_dynamics.R
        W = uav_dynamics.W
        Rt = R.T

        # Desired values (i.e., reference signals)
        [roll_d, pitch_d, yaw_d] = target
        Rd = SE3.euler_to_rotmat(roll_d, pitch_d, yaw_d)
        Wd = np.zeros(3)
        W_dot_d = np.zeros(3)

        # Attitude errors
        Rdt = Rd.T
        eR_prv = 0.5 * np.trace(np.eye(3) - Rdt @ R)
        eR = 0.5 * SE3.vee_map_3x3(Rdt @ R - Rt @ Rd)
        eW = W - Rt @ Rd @ Wd

        # Control moment (torque)
        WJW = np.cross(W, J @ W)
        M_ff = WJW - J @ (SE3.hat_map_3x3(W) @ Rt @
                          Rd @ Wd - Rt @ Rd @ W_dot_d)

        uav_ctrl_M = -self.kR * eR - self.kW * eW + M_ff

        return uav_ctrl_M


class GeometricTrackingController:
    def __init__(self, args):
        self.iterations = args.iterations
        self.dt = args.dt

        # Controller gains
        self.kx = np.array([10.0, 10.0, 12.0])
        self.kv = np.array([7.0, 7.0, 12.0])
        self.kR = np.array([10.0, 10.0, 10.0])
        self.kW = np.array([2.0, 2.0, 2.0])

        # Reset data
        self.reset()

    def reset(self):
        # Reset time index
        self.idx = 0

        # Data for plotting
        self.time_arr = np.zeros(self.iterations)
        self.eR_prv_arr = np.zeros(self.iterations)
        self.eR_arr = np.zeros((3, self.iterations))
        self.eW_arr = np.zeros((3, self.iterations))
        self.ex_arr = np.zeros((3, self.iterations))
        self.ev_arr = np.zeros((3, self.iterations))
        self.M_arr = np.zeros((3, self.iterations))
        self.f_arr = np.zeros(self.iterations)

    def compute(self, uav_dynamics: Dynamics, target):
        # States and parameters
        mass = uav_dynamics.mass
        J = uav_dynamics.J
        g = uav_dynamics.g
        x = uav_dynamics.x
        v = uav_dynamics.v
        R = uav_dynamics.R
        W = uav_dynamics.W
        Rt = R.T

        # Desired values (i.e., reference signals)
        [xd, vd, ad, yaw_d, Wd, W_dot_d] = target

        # Tracking errors
        ex = x - xd
        ev = v - vd

        # Compute desired thrust vector in world frame
        e3 = np.array([0.0, 0.0, 1.0])
        f_n = -(-self.kx * ex - self.kv * ev - mass *
                g * e3 + mass * ad)

        # Desired orientation
        b1d = np.array([np.cos(yaw_d), np.sin(yaw_d), 0.0])
        b3d = f_n / np.linalg.norm(f_n)
        b2d = np.cross(b3d, b1d)
        b1d_proj = np.cross(b2d, b3d)
        Rd = np.column_stack((b1d_proj, b2d, b3d))

        # Total thrust (scalar, body z-direction)
        f_total = np.dot(f_n, R @ e3)

        # Attitude errors
        Rdt = Rd.T
        eR_prv = 0.5 * np.trace(np.eye(3) - Rdt @ R)
        eR = 0.5 * SE3.vee_map_3x3(Rdt @ R - Rt @ Rd)
        eW = W - Rt @ Rd @ Wd

        # Control moment (torque)
        WJW = np.cross(W, J @ W)
        M_ff = WJW - J @ (SE3.hat_map_3x3(W) @ Rt @
                          Rd @ Wd - Rt @ Rd @ W_dot_d)

        uav_ctrl_M = -self.kR * eR - self.kW * eW + M_ff

        # Control force (in world frame)
        uav_ctrl_f = f_total * R @ e3

        # Record data for plotting
        self.time_arr[self.idx] = self.idx * self.dt
        self.eR_prv_arr[self.idx] = eR_prv
        self.eR_arr[:, self.idx] = eR
        self.eW_arr[:, self.idx] = eW
        self.ex_arr[:, self.idx] = ex
        self.ev_arr[:, self.idx] = ev
        self.M_arr[:, self.idx] = uav_ctrl_M
        self.f_arr[self.idx] = f_total

        # Update time index
        self.idx += 1

        return uav_ctrl_M, uav_ctrl_f

    def plot_graph(self):
        # Plot control inputs
        plt.figure("Control inputs")
        plt.subplot(4, 1, 1)
        plt.plot(self.time_arr, self.M_arr[0, :])
        plt.grid(True)
        plt.title("Control inputs")
        plt.xlabel("time [s]")
        plt.ylabel("M_x")
        plt.subplot(4, 1, 2)
        plt.plot(self.time_arr, self.M_arr[1, :])
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("M_y")
        plt.subplot(4, 1, 3)
        plt.plot(self.time_arr, self.M_arr[2, :])
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("M_z")
        plt.subplot(4, 1, 4)
        plt.plot(self.time_arr, self.f_arr[:])
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("f")

        # Plot principal rotation error angle
        plt.figure("Principal Rotation Error Angle")
        plt.plot(self.time_arr, np.rad2deg(self.eR_prv_arr))
        plt.title("Principal Rotation Error Angle")
        plt.xlabel("Time [s]")
        plt.ylabel("Angle [deg]")
        plt.grid(True)

        # Plot attitude error
        plt.figure("Attitude error (eR)")
        plt.subplot(3, 1, 1)
        plt.plot(self.time_arr, np.rad2deg(self.eR_arr[0, :]))
        plt.grid(True)
        plt.title("Attitude error (eR)")
        plt.xlabel("time [s]")
        plt.ylabel("x [deg]")
        plt.subplot(3, 1, 2)
        plt.plot(self.time_arr, np.rad2deg(self.eR_arr[1, :]))
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("y [deg]")
        plt.subplot(3, 1, 3)
        plt.plot(self.time_arr, np.rad2deg(self.eR_arr[2, :]))
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("z [deg]")

        # Plot attitude rate error
        plt.figure("Angular rate error (eW)")
        plt.subplot(3, 1, 1)
        plt.plot(self.time_arr, np.rad2deg(self.eW_arr[0, :]))
        plt.grid(True)
        plt.title("Angular rate error (eW)")
        plt.xlabel("time [s]")
        plt.ylabel("x [deg/s]")
        plt.subplot(3, 1, 2)
        plt.plot(self.time_arr, np.rad2deg(self.eW_arr[1, :]))
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("y [deg/s]")
        plt.subplot(3, 1, 3)
        plt.plot(self.time_arr, np.rad2deg(self.eW_arr[2, :]))
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("z [deg/s]")

        # Plot position error
        plt.figure("Position error")
        plt.subplot(3, 1, 1)
        plt.plot(self.time_arr, self.ex_arr[0, :])
        plt.grid(True)
        plt.title("Position error")
        plt.xlabel("time [s]")
        plt.ylabel("x [m]")
        plt.subplot(3, 1, 2)
        plt.plot(self.time_arr, self.ex_arr[1, :])
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("y [m]")
        plt.subplot(3, 1, 3)
        plt.plot(self.time_arr, self.ex_arr[2, :])
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("z [m]")

        # Plot velocity error
        plt.figure("Velocity error")
        plt.subplot(3, 1, 1)
        plt.plot(self.time_arr, self.ev_arr[0, :])
        plt.grid(True)
        plt.title("Velocity error")
        plt.xlabel("time [s]")
        plt.ylabel("x [m/s]")
        plt.subplot(3, 1, 2)
        plt.plot(self.time_arr, self.ev_arr[1, :])
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("y [m/s]")
        plt.subplot(3, 1, 3)
        plt.plot(self.time_arr, self.ev_arr[2, :])
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("z [m/s]")
