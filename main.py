import argparse
import matplotlib.pyplot as plt
import numpy as np

from dynamics import Dynamics
from geometric_control import GeometricTrackingController
from rigidbody_visualize import rigidbody_visualize
from se3_math import SE3
from trajectory_planner import TrajectoryPlanner


def greeting(dynamics, iteration_times, rpy, trajectory_type="unknown"):
    roll = np.rad2deg(rpy[0])
    pitch = np.rad2deg(rpy[1])
    yaw = np.rad2deg(rpy[2])
    print(
        f"Quadrotor simulation (iterations={iteration_times}, dt={dynamics.dt:.4f} seconds)")
    print(
        f"Trajectory type: {trajectory_type}")
    print(
        f"Initial position: ({dynamics.x[0]:.2f}m, {dynamics.x[1]:.2f}m, {dynamics.x[2]:.2f}m)")
    print(
        f"Initial attitude (Euler angles): (roll={roll:.2f}deg, pitch={pitch:.2f}deg, yaw={yaw:.2f}deg)")
    print("Start simulation (press Ctrl+C to leave)...")


def main(args):
    # Initialize quadrotor dynamics
    uav_dynamics = Dynamics(
        dt=0.001, mass=1.0, J=np.diag([0.01466, 0.01466, 0.02848]))

    # Plan desired trajectory (i.e., reference signal)
    traj_planner = TrajectoryPlanner(
        args.traj, uav_dynamics.dt, args.iterations)
    traj_planner.plan()

    # Initialize quadrotor controller
    if args.ctrl == 'GEOMETRIC_CTRL':
        controller = GeometricTrackingController(
            args, uav_dynamics, traj_planner)
    else:
        raise ValueError(f"Unknown controller: {args.ctrl}")

    # Set initial position and velocity
    uav_dynamics.set_position(traj_planner.get_position(0))
    uav_dynamics.set_velocity(traj_planner.get_velocity(0))

    # Set initial orientation (from Euler angles)
    roll = np.deg2rad(0)
    pitch = np.deg2rad(0)
    yaw = np.deg2rad(traj_planner.get_yaw(0))
    R = SE3.euler_to_rotmat(roll, pitch, yaw)
    uav_dynamics.set_rotmat(R)

    # Print simulation setup information
    greeting(uav_dynamics, args.iterations, [roll, pitch, yaw], args.traj)

    # Simulation loop
    for i in range(args.iterations):
        controller.step()

    # Plot
    controller.plot()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=20000,
                        help='Number of iterations')
    parser.add_argument('--ctrl', type=str, default='GEOMETRIC_CTRL',
                        help='Controller (GEOMETRIC_CTRL)')
    parser.add_argument('--traj', type=str, default='EIGHT',
                        help='Trajectory to track (EIGHT or CIRCLE)')
    parser.add_argument('--animate', type=str, default="yes",
                        help='3D animation of flight')
    parser.add_argument('--plot', type=str, default="yes",
                        help='Plot flight data')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    try:
        main(args)
    except KeyboardInterrupt:
        print("Stop")
