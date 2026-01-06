import argparse
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import yaml

from models.dynamics import Dynamics
from models.se3_math import NumpySE3
from control.geometric_control import GeometricMomentController
from viz.rigidbody_visualize import QuadRenderer
from gymnasium import spaces
from trajectory_planner import TrajectoryPlanner


class QuadrotorEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, args,
                 uav_dynamics=None, controller=None,
                 render_mode=None, rl_training: bool = False):
        super().__init__()
        self.args = args
        self.controller = controller
        self.iterations = args.iterations
        self.dt = args.dt
        self.render_mode = render_mode
        self.rl_training = rl_training

        # Load vehicle parameters from config file
        vehicle_path = 'configs/vehicles/' + args.vehicle_cfg
        with open(vehicle_path) as f:
            cfg = yaml.safe_load(f)
        mass = float(cfg["mass"])
        J = np.array(cfg["inertia"], dtype=np.float32)        

        # Initialize quadrotor dynamics
        if uav_dynamics == None:
            self.uav_dynamics = Dynamics(dt=args.dt, mass=mass, J=J)
        else:
            self.uav_dynamics = uav_dynamics

        # Initialize trajectory planner
        traj_planner = TrajectoryPlanner(args)
        traj_planner.plan()

        # Initialize observation space for reinforcement learning
        # position error (3x1) + velocity error (3x1) + euler angles (3x1)
        pos_vel_low = np.full(6, -np.inf, dtype=np.float32)
        pos_vel_high = np.full(6, +np.inf, dtype=np.float32)
        euler_low = np.array([-np.pi, -np.pi/2, -np.pi], dtype=np.float32)
        euler_high = np.array([+np.pi, +np.pi/2, +np.pi], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.concatenate([pos_vel_low, euler_low]),
            high=np.concatenate([pos_vel_high, euler_high]),
            dtype=np.float32
        )

        # Initialize action space for reinforcement learning
        ROLL_CTRL_MIN = -np.deg2rad(50)
        ROLL_CTRL_MAX = +np.deg2rad(50)
        PITCH_CTRL_MIN = -np.deg2rad(50)
        PITCH_CTRL_MAX = +np.deg2rad(50)
        mass = self.uav_dynamics.get_mass()
        g = self.uav_dynamics.get_gravitational_acceleration()
        hover = mass * g
        THRUST_MIN = -hover
        THRUST_MAX = +hover
        self.action_space = spaces.Box(
            low=np.array([ROLL_CTRL_MIN, PITCH_CTRL_MIN,
                         THRUST_MIN], dtype=np.float32),
            high=np.array([ROLL_CTRL_MAX, PITCH_CTRL_MAX,
                          THRUST_MAX], dtype=np.float32),
            dtype=np.float32
        )

        # Initialize moment controller for reinforcement learning
        self.moment_controller = GeometricMomentController()

        # Get random seed
        self.np_random, _ = gym.utils.seeding.np_random(None)

        # Reference signals
        self.xd = traj_planner.get_position_trajectory()  # Desired position
        self.vd = traj_planner.get_velocity_trajectory()  # Desired velocity
        self.ad = np.array([0.0, 0.0, 0.0])  # Desired acceleration
        self.yaw_d = traj_planner.get_yaw_trajectory()  # Desired yaw
        self.Wd = np.array([0.0, 0.0, 0.0])  # Desired angular velocity
        # Desired angular acceleration
        self.W_dot_d = np.array([0.0, 0.0, 0.0])

        # Initialize online renderer
        if args.renderer == 'online':
            self.viz = QuadRenderer.from_online(trajectory=self.xd)
        elif args.renderer == 'offline':
            self.viz = None
        else:
            raise ValueError(f"Unknown rendering type: {args.renderer}")

        # Reset states
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Reset time index
        self.idx = 0

        # Set random seed
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Set initial position and velocity
        self.uav_dynamics.set_position(self.xd[:, 0])
        self.uav_dynamics.set_velocity(self.vd[:, 0])

        # Set initial orientation (from Euler angles)
        roll = np.deg2rad(0)
        pitch = np.deg2rad(0)
        yaw = self.yaw_d[0]
        R = NumpySE3.euler_to_rotmat(roll, pitch, yaw)
        self.uav_dynamics.set_rotmat(R)

        # Randomize initial states
        if self.args.random_start == 'yes' or self.rl_training == True:
            self.uav_dynamics.state_randomize(self.np_random)

        # Reset controller
        if self.controller != None:
            self.controller.reset()

        # Data for plotting
        self.time_arr = np.zeros(self.iterations)
        self.accel_arr = np.zeros((3, self.iterations))
        self.vel_arr = np.zeros((3, self.iterations))
        self.pos_arr = np.zeros((3, self.iterations))
        self.R_arr = np.zeros((3, 3, self.iterations))
        self.euler_arr = np.zeros((3, self.iterations))
        self.W_dot_arr = np.zeros((3, self.iterations))
        self.W_arr = np.zeros((3, self.iterations))

        # Current desired values (i.e., reference signals)
        self.curr_xd = self.xd[:, 0]
        self.curr_vd = self.vd[:, 0]
        self.curr_ad = self.ad
        self.curr_yaw_d = self.yaw_d[0]
        self.curr_Wd = self.Wd
        self.curr_W_dot_d = self.W_dot_d

        # Return data for reinforcement learning
        obs = self.get_observation()
        return obs, {}

    def compute_reward(self):
        """Compute reward for reinforcement learning"""
        obs = self.get_observation()
        ex, ev = obs[0:3], obs[3:6]
        return -float(np.linalg.norm(ex) + 0.25*np.linalg.norm(ev))

    def get_observation(self):
        """Return observation for reinforcement learning"""
        x = self.uav_dynamics.get_position()
        v = self.uav_dynamics.get_velocity()
        R = self.uav_dynamics.get_rotmat()
        ex = x - self.curr_xd
        ev = v - self.curr_vd
        euler = NumpySE3.rotmat_to_euler(R)
        return np.concatenate([ex, ev, euler]).astype(np.float32)

    def check_terminated(self):
        """Check terminaion for reinforcement learning"""
        if self.args.ctrl != 'RL':
            return False

        obs = self.get_observation()
        ex, ev = obs[0:3], obs[3:6]
        ex_too_large = np.linalg.norm(ex) > 10.0
        ev_too_large = np.linalg.norm(ev) > 30.0
        return ex_too_large or ev_too_large

    def check_truncated(self):
        return self.idx >= self.iterations

    def update_desired_state(self):
        self.curr_xd = self.xd[:, self.idx]
        self.curr_vd = self.vd[:, self.idx]
        self.curr_ad = self.ad
        self.curr_yaw_d = self.yaw_d[self.idx]
        self.curr_Wd = self.Wd
        self.curr_W_dot_d = self.W_dot_d

    def get_desired_state(self):
        return [self.curr_xd,
                self.curr_vd,
                self.curr_ad,
                self.curr_yaw_d,
                self.curr_Wd,
                self.curr_W_dot_d]

    def execute_rl_action(self, action):
        [roll_cmd, pitch_cmd, thrust_cmd] = action
        mass = self.uav_dynamics.get_mass()
        g = self.uav_dynamics.get_gravitational_acceleration()
        R = self.uav_dynamics.get_rotmat()
        hover = mass * g
        residual = action[2]
        thrust_cmd = np.clip(hover + residual, 0.0, 3.0 * hover)
        uav_ctrl_M = self.moment_controller.run(
            self.uav_dynamics, roll_cmd, pitch_cmd, self.curr_yaw_d)
        uav_ctrl_f = thrust_cmd * R @ np.array([0.0, 0.0, 1.0])
        return [uav_ctrl_M, uav_ctrl_f]

    def step(self, action):
        """action: control input from controller or reinfocement learning"""
        uav_ctrl_M = None
        uav_ctrl_f = None

        if self.args.ctrl == 'RL':
            # For reinforcement learning, the action input is desired roll,
            # pitch, and thrust commands
            [uav_ctrl_M, uav_ctrl_f] = self.execute_rl_action(action)
        else:
            # For traditional controllers, the action input is 3x1 moment and
            # force vectors
            [uav_ctrl_M, uav_ctrl_f] = action

        # Update quadrotor dyanmics
        self.uav_dynamics.set_moment(uav_ctrl_M)
        self.uav_dynamics.set_force(uav_ctrl_f)
        self.uav_dynamics.update()

        # Record data for plotting
        self.time_arr[self.idx] = self.idx * self.dt
        self.accel_arr[:, self.idx] = self.uav_dynamics.get_acceleration()
        self.vel_arr[:, self.idx] = self.uav_dynamics.get_velocity()
        self.pos_arr[:, self.idx] = self.uav_dynamics.get_position()
        self.R_arr[:, :, self.idx] = self.uav_dynamics.get_rotmat()
        self.euler_arr[:, self.idx] = NumpySE3.rotmat_to_euler(
            self.uav_dynamics.get_rotmat())
        self.W_dot_arr[:,
                       self.idx] = self.uav_dynamics.get_angular_acceleration()
        self.W_arr[:, self.idx] = self.uav_dynamics.get_angular_velocity()

        # Update time index
        self.idx += 1

        # Update desired state from next
        truncated = self.check_truncated()
        if truncated == False:
            self.update_desired_state()

        # Return data for reinforcement learning
        obs = self.get_observation()
        reward = self.compute_reward()
        terminated = self.check_terminated()
        return obs, reward, terminated, truncated, {}

    def plot_graph(self):
        if self.controller != None:
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
        if self.args.animate == 'yes' and self.args.renderer == 'offline':
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
        enabled = self.render_mode == 'human' or self.args.renderer == 'online'
        if self.idx % skip == 0 and enabled == True:
            R = self.uav_dynamics.get_rotmat()
            x = self.uav_dynamics.get_position()
            self.viz.render(R, x)
