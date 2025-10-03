import argparse
import matplotlib.pyplot as plt
import numpy as np

from dynamics import Dynamics
from rigidbody_visualize import rigidbody_visualize
from se3_math import SE3
from trajectory_planner import TrajectoryPlanner


class GeometricTrackingController:
    def __init__(self, args, uav_dynamics: Dynamics, traj_planner: TrajectoryPlanner):
        self.args = args
        self.uav_dynamics = uav_dynamics
        self.iterations = args.iterations
        self.dt = uav_dynamics.dt

        # Controller gains
        self.kx = np.array([10.0, 10.0, 12.0])
        self.kv = np.array([7.0, 7.0, 12.0])
        self.kR = np.array([10.0, 10.0, 10.0])
        self.kW = np.array([2.0, 2.0, 2.0])

        # Controller setpoints
        self.xd = traj_planner.get_position_trajectory()  # Desired position
        self.vd = traj_planner.get_velocity_trajectory()  # Desired velocity
        # Desired acceleration
        self.a_d = np.array([0.0, 0.0, 0.0])
        self.yaw_d = traj_planner.get_yaw_trajectory()    # Desired yaw
        # Desired angular velocity
        self.Wd = np.array([0.0, 0.0, 0.0])
        # Desired angular acceleration
        self.W_dot_d = np.array([0.0, 0.0, 0.0])

        # Reset states
        self.reset()

    def reset(self):
        self.idx = 0
        self.time_arr = np.zeros(self.iterations)
        self.accel_arr = np.zeros((3, self.iterations))
        self.vel_arr = np.zeros((3, self.iterations))
        self.R_arr = np.zeros((3, 3, self.iterations))
        self.euler_arr = np.zeros((3, self.iterations))
        self.pos_arr = np.zeros((3, self.iterations))
        self.W_dot_arr = np.zeros((3, self.iterations))
        self.W_arr = np.zeros((3, self.iterations))
        self.f_arr = np.zeros(self.iterations)
        self.M_arr = np.zeros((3, self.iterations))
        self.eR_prv_arr = np.zeros((3, self.iterations))
        self.eR_arr = np.zeros((3, self.iterations))
        self.eW_arr = np.zeros((3, self.iterations))
        self.ex_arr = np.zeros((3, self.iterations))
        self.ev_arr = np.zeros((3, self.iterations))

    def step(self):
        mass = self.uav_dynamics.mass
        J = self.uav_dynamics.J
        g = self.uav_dynamics.g
        x = self.uav_dynamics.x
        v = self.uav_dynamics.v
        R = self.uav_dynamics.R
        W = self.uav_dynamics.W
        Rt = R.T

        # Tracking errors
        ex = x - self.xd[:, self.idx]
        ev = v - self.vd[:, self.idx]

        # Compute desired thrust vector in world frame
        e3 = np.array([0.0, 0.0, 1.0])
        f_n = -(-self.kx * ex - self.kv * ev - mass *
                g * e3 + mass * self.a_d)

        # Desired orientation
        b1d = np.array([np.cos(self.yaw_d[self.idx]),
                       np.sin(self.yaw_d[self.idx]), 0.0])
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
        eW = W - Rt @ Rd @ self.Wd

        # Control moment (torque)
        WJW = np.cross(W, J @ W)
        M_ff = WJW - J @ (SE3.hat_map_3x3(W) @ Rt @
                          Rd @ self.Wd - Rt @ Rd @ self.W_dot_d)

        uav_ctrl_M = -self.kR * eR - self.kW * eW + M_ff

        # Control force (in world frame)
        uav_ctrl_f = f_total * R @ e3

        # Update quadrotor dyanmics
        self.uav_dynamics.set_moment(uav_ctrl_M)
        self.uav_dynamics.set_force(uav_ctrl_f)
        self.uav_dynamics.update()

        # Record data for plotting
        self.time_arr[self.idx] = self.idx * self.dt
        self.eR_prv_arr[:, self.idx] = eR_prv
        self.eR_arr[:, self.idx] = eR
        self.eW_arr[:, self.idx] = eW
        self.accel_arr[:, self.idx] = self.uav_dynamics.a
        self.vel_arr[:, self.idx] = self.uav_dynamics.v
        self.pos_arr[:, self.idx] = self.uav_dynamics.x
        self.R_arr[:, :, self.idx] = self.uav_dynamics.R
        self.euler_arr[:, self.idx] = SE3.rotmat_to_euler(self.uav_dynamics.R)
        self.W_dot_arr[:, self.idx] = self.uav_dynamics.W_dot
        self.W_arr[:, self.idx] = self.uav_dynamics.W
        self.f_arr[self.idx] = f_total
        self.M_arr[:, self.idx] = self.uav_dynamics.M
        self.ex_arr[:, self.idx] = ex
        self.ev_arr[:, self.idx] = ev

        # Update time index
        self.idx += 1

    def plot(self):
        if self.args.plot == 'yes':
            self.plot_graph()
        if self.args.animate == 'yes':
            self.animate()
        if self.args.plot == 'yes' or self.args.animate == 'yes':
            plt.show()

    def plot_graph(self):
        # Convert radians to degrees
        eR_prv_deg = np.rad2deg(self.eR_prv_arr[0, :])

        # Plot principal rotation error angle
        plt.figure("Principal Rotation Error Angle")
        plt.plot(self.time_arr, eR_prv_deg)
        plt.title("Principal Rotation Error Angle")
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

        # Plot attitude (euler angles)
        plt.figure("Attitude (euler angles)")
        plt.subplot(3, 1, 1)
        plt.plot(self.time_arr, np.rad2deg(self.euler_arr[0, :]))
        plt.grid(True)
        plt.title("Attitude (euler angles)")
        plt.xlabel("time [s]")
        plt.ylabel("roll [deg]")
        plt.subplot(3, 1, 2)
        plt.plot(self.time_arr, np.rad2deg(self.euler_arr[1, :]))
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("pitch [deg]")
        plt.subplot(3, 1, 3)
        plt.plot(self.time_arr, np.rad2deg(self.euler_arr[2, :]), label="yaw")
        plt.plot(self.time_arr, np.rad2deg(self.yaw_d),
                 label="yaw_d", linestyle="--")
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("yaw [deg]")
        plt.legend()

        # Plot position (NED frame)
        plt.figure("Position (NED frame)")
        plt.subplot(3, 1, 1)
        plt.plot(self.time_arr, self.pos_arr[0, :], label="x[0]")
        plt.plot(self.time_arr, self.xd[0, :], label="xd[0]")
        plt.grid(True)
        plt.title("Position (NED frame)")
        plt.xlabel("time [s]")
        plt.ylabel("x [m]")
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.plot(self.time_arr, self.pos_arr[1, :], label="x[1]")
        plt.plot(self.time_arr, self.xd[1, :], label="xd[1]")
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("y [m]")
        plt.legend()
        plt.subplot(3, 1, 3)
        plt.plot(self.time_arr, -self.pos_arr[2, :], label="-x[2]")
        plt.plot(self.time_arr, -self.xd[2, :], label="-xd[2]")
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("-z [m]")
        plt.legend()

        # Plot velocity (NED frame)
        plt.figure("Velocity (NED frame)")
        plt.subplot(3, 1, 1)
        plt.plot(self.time_arr, self.vel_arr[0, :], label="v[0]")
        plt.plot(self.time_arr, self.vd[0, :], label="vd[0]")
        plt.grid(True)
        plt.title("Velocity (NED frame)")
        plt.xlabel("time [s]")
        plt.ylabel("x [m/s]")
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.plot(self.time_arr, self.vel_arr[1, :], label="v[1]")
        plt.plot(self.time_arr, self.vd[1, :], label="vd[1]")
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("y [m/s]")
        plt.legend()
        plt.subplot(3, 1, 3)
        plt.plot(self.time_arr, -self.vel_arr[2, :], label="-v[2]")
        plt.plot(self.time_arr, -self.vd[2, :], label="-vd[2]")
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("-z [m/s]")
        plt.legend()

        # Plot acceleration (NED frame)
        plt.figure("Acceleration (NED frame)")
        plt.subplot(3, 1, 1)
        plt.plot(self.time_arr, self.accel_arr[0, :])
        plt.grid(True)
        plt.title("Acceleration (NED frame)")
        plt.xlabel("time [s]")
        plt.ylabel("x [m/s^2]")
        plt.subplot(3, 1, 2)
        plt.plot(self.time_arr, self.accel_arr[1, :])
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("y [m/s^2]")
        plt.subplot(3, 1, 3)
        plt.plot(self.time_arr, -self.accel_arr[2, :])
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("-z [m/s^2]")

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

        # 2D XY trajectory comparison
        plt.figure("XY Trajectory")
        plt.plot(self.xd[0, :], self.xd[1, :],
                 label="Desired Trajectory",
                 linestyle="--", linewidth=2)
        plt.plot(self.pos_arr[0, :], self.pos_arr[1, :],
                 label="True Position", alpha=0.8)
        plt.grid(True)
        plt.title("XY Trajectory")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.axis("equal")
        plt.legend()

    def animate(self):
        rigidbody_visualize(self.pos_arr.T,
                            self.R_arr.transpose(2, 0, 1),
                            plot_size=(5, 5, 5),
                            skip=10,
                            axis_length=1.5,
                            dt=self.dt,
                            ref_traj=self.xd.T)
