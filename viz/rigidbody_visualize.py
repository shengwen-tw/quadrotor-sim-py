import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


class QuadRenderer:
    """
    Unified Online / Offline renderer.
    Use classmethods from_online() / from_offline() for explicit initialization.
    """

    #==========================#
    # Base init (shared setup) #
    #==========================#
    def __init__(self, *, plot_size=(5, 5, 5), axis_length: float = 1.0,
                 arm_ratio: float = 0.6, figsize=(6, 5), trajectory: np.ndarray = None):
        self.axis_length = float(axis_length)
        self.arm_len = float(axis_length * arm_ratio)

        # Create figure and 3D axes
        self.fig = plt.figure("3D Visualization", figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Set axis limits and labels (NED convention)
        sx, sy, sz = plot_size
        self.ax.set_xlim(-sy, sy)  # East (Y)
        self.ax.set_ylim(-sx, sx)  # North (X)
        self.ax.set_zlim(sz, -sz)  # Down (Z)
        self.ax.set_xlabel("East [m]")
        self.ax.set_ylabel("North [m]")
        self.ax.set_zlabel("Down [m]")

        # Plot reference trajectory if provided
        if trajectory is not None:
            self.ax.plot(trajectory[0, :], trajectory[1, :], trajectory[2, :],
                         color='orange', linestyle='--', label='Desired Trajectory')
            self.ax.legend()

        # Initialize body axes (x, y, z)
        (self.x_axis_line,) = self.ax.plot([], [], [], 'r-', linewidth=2)
        (self.y_axis_line,) = self.ax.plot([], [], [], 'g-', linewidth=2)
        (self.z_axis_line,) = self.ax.plot([], [], [], 'b-', linewidth=2)

        # Initialize motors and arms
        self.motor_dots = [self.ax.plot([], [], [], 'ko', markersize=4)[
            0] for _ in range(4)]
        (self.arm_line_1,) = self.ax.plot([], [], [], 'k-', linewidth=2)
        (self.arm_line_2,) = self.ax.plot([], [], [], 'k-', linewidth=2)

        self.fig.tight_layout()

        # Placeholders for offline mode
        self.mode = None
        self.pos_history = None
        self.R_history = None
        self.N = None
        self.interval_ms = None

    #==========================#
    # Classmethod constructors #
    #==========================#
    @classmethod
    def from_online(cls, *, trajectory: np.ndarray = None,
                    plot_size=(5, 5, 5), axis_length: float = 1.0,
                    arm_ratio: float = 0.6, figsize=(6, 5)):
        """Initialize renderer for online (step-by-step) updates."""
        obj = cls(plot_size=plot_size, axis_length=axis_length,
                  arm_ratio=arm_ratio, figsize=figsize, trajectory=trajectory)
        obj.mode = "online"
        return obj

    @classmethod
    def from_offline(cls, pos_history: np.ndarray, R_history: np.ndarray, dt: float,
                     *, trajectory: np.ndarray = None, plot_size=(5, 5, 5),
                     axis_length: float = 1.0, arm_ratio: float = 0.6,
                     figsize=(6, 5), skip: int = 10):
        """Initialize renderer for offline (animation) visualization."""
        obj = cls(plot_size=plot_size, axis_length=axis_length,
                  arm_ratio=arm_ratio, figsize=figsize, trajectory=trajectory)
        obj.mode = "offline"
        pos_history_t = pos_history.T
        obj.pos_history = pos_history_t[::skip]
        R_history_enu = R_history.transpose(2, 0, 1)
        obj.R_history = R_history_enu[::skip]
        obj.N = len(obj.pos_history)
        obj.interval_ms = dt * skip * 1000  # ms per frame
        return obj

    #==================#
    # Helper functions #
    #==================#
    @staticmethod
    def body_axes_world(R, p, L):
        """Return line segments for body x/y/z axes"""
        ex, ey, ez = R[:, 0], R[:, 1], R[:, 2]
        x_seg = np.stack([p, p + L*ex], axis=0)
        y_seg = np.stack([p, p + L*ey], axis=0)
        z_seg = np.stack([p, p + L*ez], axis=0)
        return x_seg, y_seg, z_seg

    def motor_world_coords(self, R, p):
        """Compute motor positions in world frame"""
        a = self.arm_len
        motor_body = np.array([
            [a,  a,  0.1],
            [-a, a,  0.1],
            [-a, -a,  0.1],
            [a, -a,  0.1],
        ])
        return (motor_body @ R.T) + p

    @staticmethod
    def arm_segments(motors):
        """Return two diagonal arm segments connecting four motors"""
        seg1 = np.stack([motors[0], motors[2]], axis=0)
        seg2 = np.stack([motors[1], motors[3]], axis=0)
        return seg1, seg2

    #=================#
    # Shared function #
    #=================#
    def close(self):
        """Close the figure"""
        plt.close(self.fig)
        self.fig = None
        self.ax = None

    #=============================#
    # Online: single frame update #
    #=============================#
    def render(self, R: np.ndarray, pos: np.ndarray):
        """Render one frame in online mode (step-by-step update)"""
        p = pos.reshape(3)

        # Update body axes
        x_seg, y_seg, z_seg = self.body_axes_world(R, p, self.axis_length)
        self.x_axis_line.set_data(x_seg[:, 0], x_seg[:, 1])
        self.x_axis_line.set_3d_properties(x_seg[:, 2])
        self.y_axis_line.set_data(y_seg[:, 0], y_seg[:, 1])
        self.y_axis_line.set_3d_properties(y_seg[:, 2])
        self.z_axis_line.set_data(z_seg[:, 0], z_seg[:, 1])
        self.z_axis_line.set_3d_properties(z_seg[:, 2])

        # Update motors and arms
        motors = self.motor_world_coords(R, p)
        for j, dot in enumerate(self.motor_dots):
            dot.set_data([motors[j, 0]], [motors[j, 1]])
            dot.set_3d_properties([motors[j, 2]])
        seg1, seg2 = self.arm_segments(motors)
        self.arm_line_1.set_data(seg1[:, 0], seg1[:, 1])
        self.arm_line_1.set_3d_properties(seg1[:, 2])
        self.arm_line_2.set_data(seg2[:, 0], seg2[:, 1])
        self.arm_line_2.set_3d_properties(seg2[:, 2])

        # Refresh canvas
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    #=================================================#
    # Offline Visualization: Replay Flight Trajectory #
    #=================================================#
    def update_frame(self, frame: int):
        """Update function for FuncAnimation in offline mode"""
        p = self.pos_history[frame]
        R = self.R_history[frame]

        # Update body axes
        x_seg, y_seg, z_seg = self.body_axes_world(R, p, self.axis_length)
        self.x_axis_line.set_data(x_seg[:, 0], x_seg[:, 1])
        self.x_axis_line.set_3d_properties(x_seg[:, 2])
        self.y_axis_line.set_data(y_seg[:, 0], y_seg[:, 1])
        self.y_axis_line.set_3d_properties(y_seg[:, 2])
        self.z_axis_line.set_data(z_seg[:, 0], z_seg[:, 1])
        self.z_axis_line.set_3d_properties(z_seg[:, 2])

        # Update motors and arms
        motors = self.motor_world_coords(R, p)
        for j, dot in enumerate(self.motor_dots):
            dot.set_data([motors[j, 0]], [motors[j, 1]])
            dot.set_3d_properties([motors[j, 2]])
        seg1, seg2 = self.arm_segments(motors)
        self.arm_line_1.set_data(seg1[:, 0], seg1[:, 1])
        self.arm_line_1.set_3d_properties(seg1[:, 2])
        self.arm_line_2.set_data(seg2[:, 0], seg2[:, 1])
        self.arm_line_2.set_3d_properties(seg2[:, 2])

        return [self.x_axis_line, self.y_axis_line, self.z_axis_line] + self.motor_dots + [self.arm_line_1, self.arm_line_2]

    def animate(self):
        """Run offline animation"""
        if self.mode != "offline":
            raise RuntimeError(
                "animate() only works in offline mode. Use from_offline(...) to initialize.")
        anim = FuncAnimation(self.fig, self.update_frame, frames=self.N,
                             interval=self.interval_ms, blit=False, repeat=False)
        plt.show()
