import numpy as np
import yaml


class TrajectoryPlanner:
    def __init__(self, args):
        self.traj_type = args.traj
        self.plan_yaw_traj = args.plan_yaw_traj
        self.dt = args.dt
        self.iterations = args.iterations

        self.xd = np.zeros((3, self.iterations))
        self.vd = np.zeros((3, self.iterations))
        self.yaw_d = np.zeros(self.iterations)

        # Load trajectory setting
        cfg_file = None
        if self.traj_type == "CIRCLE" or self.traj_type == "EIGHT":
            cfg_file = args.motion_cfg
        elif self.traj_type == "HOVERING":
            cfg_file = 'hover.yaml'
        cfg_full_path = 'configs/trajectory/' + cfg_file

        with open(cfg_full_path) as f:
            self.cfg = yaml.safe_load(f)["trajectory"]

    def plan(self):
        if self.plan_yaw_traj == "yes":
            self.plan_yaw_trajectory(yaw_rate=self.cfg['yaw']['yaw_rate'])

        if self.traj_type == "CIRCLE":
            self.plan_circular_trajectory(
                radius=self.cfg['circle']['radius'],
                circum_rate=self.cfg['circle']['circum_rate'],
            )
        elif self.traj_type == "EIGHT":
            self.plan_figure8_trajectory(
                A=self.cfg['eight']['A'], B=self.cfg['eight']['B'],
                a=self.cfg['eight']['a'], b=self.cfg['eight']['b'],
            )
        elif self.traj_type == "HOVERING":
            self.plan_hovering_trajectory(
                position=np.array([self.cfg['hovering']['x'],
                                   self.cfg['hovering']['y'],
                                   self.cfg['hovering']['z']]),
            )
        else:
            raise ValueError(f"Unknown trajectory type: {self.traj_type}")

    def get_position_trajectory(self):
        return self.xd

    def get_velocity_trajectory(self):
        return self.vd

    def get_yaw_trajectory(self):
        return self.yaw_d

    def get_position(self, idx: int):
        return self.xd[:, idx]

    def get_velocity(self, idx: int):
        return self.vd[:, idx]

    def get_yaw(self, idx: int):
        return self.yaw_d[idx]

    def plan_yaw_trajectory(self, yaw_rate):
        for i in range(1, self.iterations):
            w = 2 * np.pi * yaw_rate
            self.yaw_d[i] = self.yaw_d[i - 1] + w * self.dt
            if self.yaw_d[i] > np.pi:
                self.yaw_d[i] -= 2 * np.pi

    def plan_hovering_trajectory(self, position):
        for i in range(self.iterations):
            t = i * self.dt

            # Position
            self.xd[0, i] = position[0]
            self.xd[1, i] = position[1]
            self.xd[2, i] = position[2]

            # Velocity
            self.vd[0, i] = 0.0
            self.vd[1, i] = 0.0
            self.vd[2, i] = 0.0

    def plan_circular_trajectory(self, radius, circum_rate):
        w = 2 * np.pi * circum_rate

        for i in range(self.iterations):
            t = i * self.dt

            # Position
            self.xd[0, i] = radius * np.cos(w * t)
            self.xd[1, i] = radius * np.sin(w * t)
            self.xd[2, i] = -1.0

            # Velocity
            self.vd[0, i] = radius * w * -np.sin(w * t)
            self.vd[1, i] = radius * w * np.cos(w * t)
            self.vd[2, i] = 0.0

    def plan_figure8_trajectory(self, A, B, a, b):
        """
        Generate a Lissajous-type figure-8 trajectory.

        Args:
            A (float): Amplitude in x-direction
            B (float): Amplitude in y-direction
            a (int): Frequency multiplier in x-direction
            b (int): Frequency multiplier in y-direction
            yaw_rate (float): Desired yaw rate in Hz
            dt (float): Time step
            iteration_times (int): Number of time steps
        """
        omega = 2 * np.pi  # 1 Hz base angular frequency

        for i in range(self.iterations):
            t = i * self.dt

            # Position
            self.xd[0, i] = A * np.sin(a * omega * t)
            self.xd[1, i] = B * np.sin(b * omega * t)
            self.xd[2, i] = -1.0

            # Velocity
            self.vd[0, i] = A * a * omega * np.cos(a * omega * t)
            self.vd[1, i] = B * b * omega * np.cos(b * omega * t)
            self.vd[2, i] = 0.0
