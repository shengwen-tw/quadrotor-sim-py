import argparse
import numpy as np
import matplotlib.pyplot as plt

from dynamics import Dynamics
from se3_math import SE3
from rigidbody_visualize import rigidbody_visualize


def greeting(dynamics, iteration_times, init_attitude_rad, trajectory_type="unknown"):
    roll = np.rad2deg(init_attitude_rad[0])
    pitch = np.rad2deg(init_attitude_rad[1])
    yaw = np.rad2deg(init_attitude_rad[2])
    print(
        f"Quadrotor simulation (iterations={iteration_times}, dt={dynamics.dt:.4f} seconds)")
    print(
        f"Trajectory type: {trajectory_type}")
    print(
        f"Initial position: ({dynamics.x[0]:.2f}m, {dynamics.x[1]:.2f}m, {dynamics.x[2]:.2f}m)")
    print(
        f"Initial attitude (Euler angles): (roll={roll:.2f}deg, pitch={pitch:.2f}deg, yaw={yaw:.2f}deg)")
    print("Start simulation...")


def plan_circular_trajectory(radius, circum_rate, yaw_rate, dt, iteration_times):
    xd = np.zeros((3, iteration_times))
    vd = np.zeros((3, iteration_times))
    yaw_d = np.zeros(iteration_times)

    w = 2 * np.pi * circum_rate

    for i in range(iteration_times):
        t = i * dt

        # Position
        xd[0, i] = radius * np.cos(w * t)
        xd[1, i] = radius * np.sin(w * t)
        xd[2, i] = -1.0

        # Velocity
        vd[0, i] = radius * w * -np.sin(w * t)
        vd[1, i] = radius * w * np.cos(w * t)
        vd[2, i] = 0.0

        # Yaw
        if i == 0:
            yaw_d[i] = 0.0
        else:
            yaw_d[i] = yaw_d[i - 1] + yaw_rate * dt * 2 * np.pi
            if yaw_d[i] > np.pi:
                yaw_d[i] -= 2 * np.pi

    return xd, vd, yaw_d


def plan_figure8_trajectory(A, B, a, b, yaw_rate, dt, iteration_times):
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
    xd = np.zeros((3, iteration_times))
    vd = np.zeros((3, iteration_times))
    yaw_d = np.zeros(iteration_times)

    omega = 2 * np.pi  # 1 Hz base angular frequency

    for i in range(iteration_times):
        t = i * dt

        # Position
        xd[0, i] = A * np.sin(a * omega * t)
        xd[1, i] = B * np.sin(b * omega * t)
        xd[2, i] = -1.0

        # Velocity
        vd[0, i] = A * a * omega * np.cos(a * omega * t)
        vd[1, i] = B * b * omega * np.cos(b * omega * t)
        vd[2, i] = 0.0

        # Yaw
        if i == 0:
            yaw_d[i] = 0.0
        else:
            yaw_d[i] = yaw_d[i - 1] + yaw_rate * dt * 2 * np.pi
            if yaw_d[i] > np.pi:
                yaw_d[i] -= 2 * np.pi

    return xd, vd, yaw_d


def main(args):
    ITERATION_TIMES = 20000

    math = SE3()

    # Create UAV dynamics object
    uav_dynamics = Dynamics(dt=0.001, mass=1.0, J=np.zeros((3, 3)))

    # Initialize UAV state
    uav_dynamics.a = np.array([0.0, 0.0, 0.0])      # Linear acceleration
    uav_dynamics.v = np.array([0.0, 0.0, 0.0])      # Linear velocity
    uav_dynamics.x = np.array([0.0, 0.0, 0.0])      # Position
    uav_dynamics.W = np.array([0.0, 0.0, 0.0])      # Angular velocity
    uav_dynamics.W_dot = np.array([0.0, 0.0, 0.0])  # Angular acceleration
    uav_dynamics.f = np.array([0.0, 0.0, 0.0])      # Control force
    uav_dynamics.M = np.array([0.0, 0.0, 0.0])      # Control moment

    # Set initial orientation (from Euler angles)
    roll = np.deg2rad(0)
    pitch = np.deg2rad(0)
    yaw = np.deg2rad(0)
    init_attitude = np.array([roll, pitch, yaw])
    uav_dynamics.R = math.euler_to_rotmat(roll, pitch, yaw)

    # Set inertia matrix
    uav_dynamics.J = np.array([
        [0.01466, 0.0,     0.0],
        [0.0,     0.01466, 0.0],
        [0.0,     0.0,     0.02848]
    ])

    # Set flight trajectory
    trajectory_type = "figure8"  # "circle" or "figure8"

    # Controller gains
    kx = np.array([10.0, 10.0, 12.0])
    kv = np.array([7.0, 7.0, 12.0])
    kR = np.array([10.0, 10.0, 10.0])
    kW = np.array([2.0, 2.0, 2.0])

    # Controller setpoints
    xd = np.zeros((3, ITERATION_TIMES))    # Desired position
    vd = np.zeros((3, ITERATION_TIMES))    # Desired velocity
    a_d = np.array([0.0, 0.0, 0.0])        # Desired acceleration
    yaw_d = np.zeros(ITERATION_TIMES)      # Desired yaw
    Wd = np.array([0.0, 0.0, 0.0])         # Desired angular velocity
    W_dot_d = np.array([0.0, 0.0, 0.0])    # Desired angular acceleration

    # =================
    #   Path Planning
    # =================

    if args.traj == "CIRCLE":
        xd, vd, yaw_d = plan_circular_trajectory(
            radius=3.0,
            circum_rate=0.125,
            yaw_rate=0.05,
            dt=uav_dynamics.dt,
            iteration_times=ITERATION_TIMES
        )
    elif args.traj == "EIGHT":
        xd, vd, yaw_d = plan_figure8_trajectory(
            A=3.0, B=3.0,
            a=0.1, b=0.2,  # 1:2 ratio
            yaw_rate=0.05,
            dt=uav_dynamics.dt,
            iteration_times=ITERATION_TIMES
        )
    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")

    # Set initial UAV state to match the first point of trajectory
    uav_dynamics.x = xd[:, 0].copy()
    uav_dynamics.v = vd[:, 0].copy()

    # Print simulation setup information
    greeting(uav_dynamics, ITERATION_TIMES,
             init_attitude, trajectory_type)

    # ==========================================
    #   Preallocate arrays for simulation data
    # ==========================================

    time_arr = np.zeros(ITERATION_TIMES)
    accel_arr = np.zeros((3, ITERATION_TIMES))
    vel_arr = np.zeros((3, ITERATION_TIMES))
    R_arr = np.zeros((3, 3, ITERATION_TIMES))
    euler_arr = np.zeros((3, ITERATION_TIMES))
    pos_arr = np.zeros((3, ITERATION_TIMES))
    W_dot_arr = np.zeros((3, ITERATION_TIMES))
    W_arr = np.zeros((3, ITERATION_TIMES))
    f_arr = np.zeros(ITERATION_TIMES)
    M_arr = np.zeros((3, ITERATION_TIMES))
    eR_prv_arr = np.zeros((3, ITERATION_TIMES))
    eR_arr = np.zeros((3, ITERATION_TIMES))
    eW_arr = np.zeros((3, ITERATION_TIMES))
    ex_arr = np.zeros((3, ITERATION_TIMES))
    ev_arr = np.zeros((3, ITERATION_TIMES))

    for i in range(ITERATION_TIMES):
        # ===========================
        # 1. Update System Dynamics
        # ===========================
        uav_dynamics.update()

        # =================================
        # 2. Geometry Tracking Controller
        # =================================

        # Tracking errors
        ex = uav_dynamics.x - xd[:, i]
        ev = uav_dynamics.v - vd[:, i]

        # Compute desired thrust vector in world frame
        e3 = np.array([0.0, 0.0, 1.0])
        f_n = -(-kx * ex - kv * ev - uav_dynamics.mass *
                uav_dynamics.g * e3 + uav_dynamics.mass * a_d)

        # Desired orientation
        b1d = np.array([np.cos(yaw_d[i]), np.sin(yaw_d[i]), 0.0])
        b3d = f_n / np.linalg.norm(f_n)
        b2d = np.cross(b3d, b1d)
        b1d_proj = np.cross(b2d, b3d)
        Rd = np.column_stack((b1d_proj, b2d, b3d))

        # Total thrust (scalar, body z-direction)
        f_total = np.dot(f_n, uav_dynamics.R @ e3)

        # Attitude errors
        Rt = uav_dynamics.R.T
        Rdt = Rd.T
        eR_prv = 0.5 * np.trace(np.eye(3) - Rdt @ uav_dynamics.R)
        eR = 0.5 * SE3.vee_map_3x3(Rdt @ uav_dynamics.R - Rt @ Rd)
        eW = uav_dynamics.W - Rt @ Rd @ Wd

        # Control moment (torque)
        W = uav_dynamics.W
        J = uav_dynamics.J
        WJW = np.cross(W, J @ W)
        M_ff = WJW - J @ (SE3.hat_map_3x3(W) @ Rt @
                          Rd @ Wd - Rt @ Rd @ W_dot_d)

        uav_ctrl_M = -kR * eR - kW * eW + M_ff

        # Control force (in world frame)
        uav_ctrl_f = f_total * uav_dynamics.R @ e3

        # Apply control input to UAV model
        uav_dynamics.M = uav_ctrl_M
        uav_dynamics.f = uav_ctrl_f

        # =============================
        # 3. Record Data for Plotting
        # =============================

        time_arr[i] = i * uav_dynamics.dt
        eR_prv_arr[:, i] = eR_prv
        eR_arr[:, i] = eR
        eW_arr[:, i] = eW
        accel_arr[:, i] = uav_dynamics.a
        vel_arr[:, i] = uav_dynamics.v
        pos_arr[:, i] = uav_dynamics.x
        R_arr[:, :, i] = uav_dynamics.R
        euler_arr[:, i] = SE3.rotmat_to_euler(uav_dynamics.R)
        W_dot_arr[:, i] = uav_dynamics.W_dot
        W_arr[:, i] = uav_dynamics.W
        f_arr[i] = f_total
        M_arr[:, i] = uav_dynamics.M
        ex_arr[:, i] = ex
        ev_arr[:, i] = ev

    # Convert radians to degrees
    eR_prv_deg = np.rad2deg(eR_prv_arr[0, :])

    # Plot principal rotation error angle
    plt.figure("Principal Rotation Error Angle")
    plt.plot(time_arr, eR_prv_deg)
    plt.title("Principal Rotation Error Angle")
    plt.title("Principal Rotation Error Angle")
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [deg]")
    plt.grid(True)

    # Plot attitude error
    plt.figure("Attitude error (eR)")
    plt.subplot(3, 1, 1)
    plt.plot(time_arr, np.rad2deg(eR_arr[0, :]))
    plt.grid(True)
    plt.title("Attitude error (eR)")
    plt.xlabel("time [s]")
    plt.ylabel("x [deg]")
    plt.subplot(3, 1, 2)
    plt.plot(time_arr, np.rad2deg(eR_arr[1, :]))
    plt.grid(True)
    plt.xlabel("time [s]")
    plt.ylabel("y [deg]")
    plt.subplot(3, 1, 3)
    plt.plot(time_arr, np.rad2deg(eR_arr[2, :]))
    plt.grid(True)
    plt.xlabel("time [s]")
    plt.ylabel("z [deg]")

    # Plot attitude rate error
    plt.figure("Angular rate error (eW)")
    plt.subplot(3, 1, 1)
    plt.plot(time_arr, np.rad2deg(eW_arr[0, :]))
    plt.grid(True)
    plt.title("Angular rate error (eW)")
    plt.xlabel("time [s]")
    plt.ylabel("x [deg/s]")
    plt.subplot(3, 1, 2)
    plt.plot(time_arr, np.rad2deg(eW_arr[1, :]))
    plt.grid(True)
    plt.xlabel("time [s]")
    plt.ylabel("y [deg/s]")
    plt.subplot(3, 1, 3)
    plt.plot(time_arr, np.rad2deg(eW_arr[2, :]))
    plt.grid(True)
    plt.xlabel("time [s]")
    plt.ylabel("z [deg/s]")

    # Plot attitude (euler angles)
    plt.figure("Attitude (euler angles)")
    plt.subplot(3, 1, 1)
    plt.plot(time_arr, np.rad2deg(euler_arr[0, :]))
    plt.grid(True)
    plt.title("Attitude (euler angles)")
    plt.xlabel("time [s]")
    plt.ylabel("roll [deg]")
    plt.subplot(3, 1, 2)
    plt.plot(time_arr, np.rad2deg(euler_arr[1, :]))
    plt.grid(True)
    plt.xlabel("time [s]")
    plt.ylabel("pitch [deg]")
    plt.subplot(3, 1, 3)
    plt.plot(time_arr, np.rad2deg(euler_arr[2, :]), label="yaw")
    plt.plot(time_arr, np.rad2deg(yaw_d), label="yaw_d", linestyle="--")
    plt.grid(True)
    plt.xlabel("time [s]")
    plt.ylabel("yaw [deg]")
    plt.legend()

    # Plot position (NED frame)
    plt.figure("Position (NED frame)")
    plt.subplot(3, 1, 1)
    plt.plot(time_arr, pos_arr[0, :], label="x")
    plt.plot(time_arr, xd[0, :], label="x_d")
    plt.grid(True)
    plt.title("Position (NED frame)")
    plt.xlabel("time [s]")
    plt.ylabel("x [m]")
    plt.subplot(3, 1, 2)
    plt.plot(time_arr, pos_arr[1, :], label="y")
    plt.plot(time_arr, xd[1, :], label="y_d")
    plt.grid(True)
    plt.xlabel("time [s]")
    plt.ylabel("y [m]")
    plt.subplot(3, 1, 3)
    plt.plot(time_arr, -pos_arr[2, :], label="-z")
    plt.plot(time_arr, -xd[2, :], label="-z_d")
    plt.grid(True)
    plt.xlabel("time [s]")
    plt.ylabel("-z [m]")
    plt.legend()

    # Plot velocity (NED frame)
    plt.figure("Velocity (NED frame)")
    plt.subplot(3, 1, 1)
    plt.plot(time_arr, vel_arr[0, :], label="x")
    plt.plot(time_arr, vd[0, :], label="x_d")
    plt.grid(True)
    plt.title("Velocity (NED frame)")
    plt.xlabel("time [s]")
    plt.ylabel("x [m/s]")
    plt.subplot(3, 1, 2)
    plt.plot(time_arr, vel_arr[1, :], label="y")
    plt.plot(time_arr, vd[1, :], label="y_d")
    plt.grid(True)
    plt.xlabel("time [s]")
    plt.ylabel("y [m/s]")
    plt.subplot(3, 1, 3)
    plt.plot(time_arr, -vel_arr[2, :], label="-z")
    plt.plot(time_arr, -vd[2, :], label="-z_d")
    plt.grid(True)
    plt.xlabel("time [s]")
    plt.ylabel("-z [m/s]")
    plt.legend()

    # Plot acceleration (NED frame)
    plt.figure("Acceleration (NED frame)")
    plt.subplot(3, 1, 1)
    plt.plot(time_arr, accel_arr[0, :])
    plt.grid(True)
    plt.title("Acceleration (NED frame)")
    plt.xlabel("time [s]")
    plt.ylabel("x [m/s^2]")
    plt.subplot(3, 1, 2)
    plt.plot(time_arr, accel_arr[1, :])
    plt.grid(True)
    plt.xlabel("time [s]")
    plt.ylabel("y [m/s^2]")
    plt.subplot(3, 1, 3)
    plt.plot(time_arr, -accel_arr[2, :])
    plt.grid(True)
    plt.xlabel("time [s]")
    plt.ylabel("-z [m/s^2]")

    # Plot position error
    plt.figure("Position error")
    plt.subplot(3, 1, 1)
    plt.plot(time_arr, ex_arr[0, :])
    plt.grid(True)
    plt.title("Position error")
    plt.xlabel("time [s]")
    plt.ylabel("x [m]")
    plt.subplot(3, 1, 2)
    plt.plot(time_arr, ex_arr[1, :])
    plt.grid(True)
    plt.xlabel("time [s]")
    plt.ylabel("y [m]")
    plt.subplot(3, 1, 3)
    plt.plot(time_arr, ex_arr[2, :])
    plt.grid(True)
    plt.xlabel("time [s]")
    plt.ylabel("z [m]")

    # Plot velocity error
    plt.figure("Velocity error")
    plt.subplot(3, 1, 1)
    plt.plot(time_arr, ev_arr[0, :])
    plt.grid(True)
    plt.title("Velocity error")
    plt.xlabel("time [s]")
    plt.ylabel("x [m/s]")
    plt.subplot(3, 1, 2)
    plt.plot(time_arr, ev_arr[1, :])
    plt.grid(True)
    plt.xlabel("time [s]")
    plt.ylabel("y [m/s]")
    plt.subplot(3, 1, 3)
    plt.plot(time_arr, ev_arr[2, :])
    plt.grid(True)
    plt.xlabel("time [s]")
    plt.ylabel("z [m/s]")

    # Plot control inputs
    plt.figure("Control inputs")
    plt.subplot(4, 1, 1)
    plt.plot(time_arr, M_arr[0, :])
    plt.grid(True)
    plt.title("Control inputs")
    plt.xlabel("time [s]")
    plt.ylabel("M_x")
    plt.subplot(4, 1, 2)
    plt.plot(time_arr, M_arr[1, :])
    plt.grid(True)
    plt.xlabel("time [s]")
    plt.ylabel("M_y")
    plt.subplot(4, 1, 3)
    plt.plot(time_arr, M_arr[2, :])
    plt.grid(True)
    plt.xlabel("time [s]")
    plt.ylabel("M_z")
    plt.subplot(4, 1, 4)
    plt.plot(time_arr, f_arr[:])
    plt.grid(True)
    plt.xlabel("time [s]")
    plt.ylabel("f")

    # 2D XY trajectory comparison
    plt.figure("XY Trajectory")
    plt.plot(xd[0, :], xd[1, :], label="Desired Trajectory",
             linestyle="--", linewidth=2)
    plt.plot(pos_arr[0, :], pos_arr[1, :], label="True Position", alpha=0.8)
    plt.grid(True)
    plt.title("XY Trajectory")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.axis("equal")
    plt.legend()

    print("Press Ctrl+C to leave...")
    if args.animate == True:
        rigidbody_visualize(pos_arr.T, R_arr.transpose(2, 0, 1),
                            plot_size=(5, 5, 5),
                            skip=10,
                            axis_length=1.5,
                            dt=uav_dynamics.dt,
                            ref_traj=xd.T)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj', type=str, default='EIGHT', help='Trajectory to track (EIGHT or CIRCLE)')
    parser.add_argument('--animate', type=bool, default=True, help='3D animation of flight')
    parser.add_argument('--plot', type=bool, default=True, help='Plot flight data')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    try:
        main(args)
    except KeyboardInterrupt:
        print("Stop")
