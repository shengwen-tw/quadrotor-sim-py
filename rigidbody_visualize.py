import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def rigidbody_visualize(pos_history: np.ndarray, R_history: np.ndarray,
                        *, plot_size=(10, 10, 10), skip: int = 10,
                        axis_length: float = 1.0, dt: float,
                        ref_traj: np.ndarray = None):
    """
    Visualize 3D rigid body orientation and position in NED frame with quadrotor shape.
    """
    pos_history = pos_history[::skip]
    R_history = R_history[::skip]
    N = len(pos_history)

    sx, sy, sz = plot_size
    interval_ms = dt * skip * 1000  # Display time per frame in milliseconds

    fig = plt.figure("3D Visualization")
    ax = fig.add_subplot(111, projection='3d')

    # Set NED axis
    ax.set_xlim(-sy, sy)  # East (Y)
    ax.set_ylim(-sx, sx)  # North (X)
    ax.set_zlim(sz, -sz)  # Down (Z)
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_zlabel("Down [m]")

    # Plot reference trajectory if provided
    if ref_traj is not None:
        ref_traj = ref_traj[::skip]
        ax.plot(ref_traj[:, 0], ref_traj[:, 1], ref_traj[:, 2],
                color='orange', linestyle='--', label='Desired Trajectory')
        ax.legend()

    # Initial body axes quivers
    colors = ("r", "g", "b")
    quivers = [ax.quiver(0, 0, 0, 0, 0, 0, length=axis_length, color=c)
               for c in colors]

    motor_dots = [ax.plot([], [], [], 'ko')[0] for _ in range(4)]
    arm_length = axis_length * 0.6
    motor_body_coords = np.array([
        [arm_length,  arm_length,  0.1],
        [-arm_length,  arm_length,  0.1],
        [-arm_length, -arm_length,  0.1],
        [arm_length, -arm_length,  0.1],
    ])
    line1, = ax.plot([], [], [], 'k-')
    line2, = ax.plot([], [], [], 'k-')

    def update(frame):
        p = pos_history[frame]
        R = R_history[frame].T

        for k in range(3):
            quivers[k].remove()
            quivers[k] = ax.quiver(
                p[0], p[1], p[2],
                R[0, k], R[1, k], R[2, k],
                length=axis_length,
                color=colors[k]
            )

        motor_world_coords = (R @ motor_body_coords.T).T + p
        for j in range(4):
            motor_dots[j].set_data(
                motor_world_coords[j, 0], motor_world_coords[j, 1])
            motor_dots[j].set_3d_properties(motor_world_coords[j, 2])

        line1.set_data([motor_world_coords[0, 0], motor_world_coords[2, 0]],
                       [motor_world_coords[0, 1], motor_world_coords[2, 1]])
        line1.set_3d_properties(
            [motor_world_coords[0, 2], motor_world_coords[2, 2]])

        line2.set_data([motor_world_coords[1, 0], motor_world_coords[3, 0]],
                       [motor_world_coords[1, 1], motor_world_coords[3, 1]])
        line2.set_3d_properties(
            [motor_world_coords[1, 2], motor_world_coords[3, 2]])

        return quivers + motor_dots + [line1, line2]

    anim = FuncAnimation(fig, update, frames=N,
                         interval=interval_ms, blit=False, repeat=False)

    plt.show()
