import argparse
import matplotlib.pyplot as plt
import numpy as np

from dynamics import Dynamics
from rigidbody_visualize import QuadRenderer
from se3_math import SE3
from trajectory_planner import TrajectoryPlanner


class QuadrotorEnv:
    def __init__(self, args, uav_dynamics: Dynamics, controller, traj_planner: TrajectoryPlanner):
        self.args = args
        self.uav_dynamics = uav_dynamics
        self.controller = controller
        self.iterations = args.iterations
        self.dt = args.dt

        # Reference signals
        self.xd = traj_planner.get_position_trajectory()  # Desired position
        self.vd = traj_planner.get_velocity_trajectory()  # Desired velocity
        self.ad = np.array([0.0, 0.0, 0.0])  # Desired acceleration
        self.yaw_d = traj_planner.get_yaw_trajectory()  # Desired yaw
        self.Wd = np.array([0.0, 0.0, 0.0])  # Desired angular velocity
        # Desired angular acceleration
        self.W_dot_d = np.array([0.0, 0.0, 0.0])

        # Initialize online renderer
        if args.render == 'online':
            self.viz = QuadRenderer.from_online(self.xd)
        elif args.render == 'offline':
            self.viz = None
        else:
            raise ValueError(f"Unknown rendering type: {args.render}")

        # Reset states
        self.reset()

    def reset(self):
        # Reset time index
        self.idx = 0

        # Data for plotting
        self.time_arr = np.zeros(self.iterations)
        self.accel_arr = np.zeros((3, self.iterations))
        self.vel_arr = np.zeros((3, self.iterations))
        self.pos_arr = np.zeros((3, self.iterations))
        self.R_arr = np.zeros((3, 3, self.iterations))
        self.euler_arr = np.zeros((3, self.iterations))
        self.W_dot_arr = np.zeros((3, self.iterations))
        self.W_arr = np.zeros((3, self.iterations))

    def step(self):
        # Compute control input
        uav_ctrl_M, uav_ctrl_f = self.controller.compute(self.uav_dynamics,
                                                         self.xd[:, self.idx],
                                                         self.vd[:, self.idx],
                                                         self.ad,
                                                         self.yaw_d[self.idx],
                                                         self.Wd,
                                                         self.W_dot_d)

        # Update quadrotor dyanmics
        self.uav_dynamics.set_moment(uav_ctrl_M)
        self.uav_dynamics.set_force(uav_ctrl_f)
        self.uav_dynamics.update()

        # Record data for plotting
        self.time_arr[self.idx] = self.idx * self.dt
        self.accel_arr[:, self.idx] = self.uav_dynamics.a
        self.vel_arr[:, self.idx] = self.uav_dynamics.v
        self.pos_arr[:, self.idx] = self.uav_dynamics.x
        self.R_arr[:, :, self.idx] = self.uav_dynamics.R
        self.euler_arr[:, self.idx] = SE3.rotmat_to_euler(self.uav_dynamics.R)
        self.W_dot_arr[:, self.idx] = self.uav_dynamics.W_dot
        self.W_arr[:, self.idx] = self.uav_dynamics.W

        # Update time index
        self.idx += 1

    def plot_graph(self):
        self.controller.plot_graph()

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

    def plot(self):
        if self.args.plot == 'yes':
            self.plot_graph()
        if self.args.animate == 'yes' and self.args.render == 'offline':
            self.render_offline()
        if self.args.plot == 'yes' or self.args.animate == 'yes':
            plt.show()

    def render_offline(self):
        self.viz = QuadRenderer.from_offline(self.pos_arr, self.R_arr,
                                             dt=self.dt,
                                             trajectory=self.xd)
        self.viz.animate()

    def render(self):
        skip = 10
        if self.idx % skip == 0 and self.args.render == 'online':
            self.viz.render(self.uav_dynamics.R, self.uav_dynamics.x)
