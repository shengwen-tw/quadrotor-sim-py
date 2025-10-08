import numpy as np


class TrajectoryPlanner:
    def __init__(self, args):
        self.traj_type = args.traj
        self.dt = args.dt
        self.iterations = args.iterations

        self.xd = np.zeros((3, self.iterations))
        self.vd = np.zeros((3, self.iterations))
        self.yaw_d = np.zeros(self.iterations)

    def plan(self):
        if self.traj_type == "CIRCLE":
            self.plan_circular_trajectory(
                radius=3.0,
                circum_rate=0.125,
                yaw_rate=0.05
            )
        elif self.traj_type == "EIGHT":
            self.plan_figure8_trajectory(
                A=3.0, B=3.0,
                a=0.1, b=0.2,  # 1:2 ratio
                yaw_rate=0.05
            )
        elif self.traj_type == "HOVERING":
            self.plan_hovering_trajectory(
                position=np.array([1, 2, -3]),
                yaw_rate=0.05
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

    def plan_hovering_trajectory(self, position, yaw_rate, plan_yaw=False):
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

            # Skip yaw planning
            if plan_yaw == False:
                continue

            # Yaw
            if i == 0:
                self.yaw_d[i] = 0.0
            else:
                self.yaw_d[i] = self.yaw_d[i - 1] + \
                    yaw_rate * self.dt * 2 * np.pi
                if self.yaw_d[i] > np.pi:
                    self.yaw_d[i] -= 2 * np.pi

    def plan_circular_trajectory(self, radius, circum_rate, yaw_rate):
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

            # Yaw
            if i == 0:
                self.yaw_d[i] = 0.0
            else:
                self.yaw_d[i] = self.yaw_d[i - 1] + \
                    yaw_rate * self.dt * 2 * np.pi
                if self.yaw_d[i] > np.pi:
                    self.yaw_d[i] -= 2 * np.pi

    def plan_figure8_trajectory(self, A, B, a, b, yaw_rate):
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

            # Yaw
            if i == 0:
                self.yaw_d[i] = 0.0
            else:
                self.yaw_d[i] = self.yaw_d[i - 1] + \
                    yaw_rate * self.dt * 2 * np.pi
                if self.yaw_d[i] > np.pi:
                    self.yaw_d[i] -= 2 * np.pi
