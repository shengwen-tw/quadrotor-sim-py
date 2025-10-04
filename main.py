import argparse
import numpy as np

from dynamics import Dynamics
from geometric_control import GeometricTrackingController
from quadrotor import QuadrotorEnv
from se3_math import SE3
from trajectory_planner import TrajectoryPlanner


def greeting(dynamics, iteration_times, trajectory_type):
    rpy = np.rad2deg(SE3.rotmat_to_euler(dynamics.R))
    W = np.rad2deg(dynamics.W)
    print(
        f"Quadrotor simulation (iterations={iteration_times}, dt={dynamics.dt:.4f} seconds)")
    print(
        f"Trajectory type: {trajectory_type}")
    print(
        f"Initial position: ({dynamics.x[0]:.2f}m, {dynamics.x[1]:.2f}m, {dynamics.x[2]:.2f}m)")
    print(
        f"Initial velocity: ({dynamics.v[0]:.2f}m/s, {dynamics.v[1]:.2f}m/s, {dynamics.v[2]:.2f}m/s)")
    print(
        f"Initial attitude: (roll={rpy[0]:.2f}deg, pitch={rpy[1]:.2f}deg, yaw={rpy[2]:.2f}deg)")
    print(
        f"Initial angular velocity: ({W[0]:.2f}deg/s, {W[1]:.2f}deg/s, {W[2]:.2f}deg/s)")
    print("Start simulation (press Ctrl+C to leave)...")


def main(args):
    # Initialize quadrotor dynamics
    uav_dynamics = Dynamics(
        dt=args.dt, mass=1.0, J=np.diag([0.01466, 0.01466, 0.02848]))

    # Initialize trajectory planner
    traj_planner = TrajectoryPlanner(args)
    traj_planner.plan()

    # Initialize quadrotor controller
    controller = None
    if args.ctrl == 'GEOMETRIC_CTRL':
        controller = GeometricTrackingController(args)
    else:
        raise ValueError(f"Unknown controller: {args.ctrl}")

    # Initialize quadrotor environment
    env = QuadrotorEnv(args, uav_dynamics, controller, traj_planner)

    # Print simulation setup information
    greeting(uav_dynamics, args.iterations, args.traj)

    # Simulation loop
    for i in range(args.iterations):
        target = env.next_target()
        action = controller.compute(uav_dynamics, target)
        env.step(action)
        env.render()

    # Plot
    env.plot()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dt', type=float, default=0.001,
                        help='Time period for simulation')
    parser.add_argument('--iterations', type=int, default=20000,
                        help='Number of iterations')
    parser.add_argument('--ctrl', type=str, default='GEOMETRIC_CTRL',
                        help='Controller (GEOMETRIC_CTRL)')
    parser.add_argument('--traj', type=str, default='EIGHT',
                        help='Trajectory to track (EIGHT, CIRCLE or HOVERING)')
    parser.add_argument('--random_start', type=str, default='no',
                        help='Random initial state')
    parser.add_argument('--renderer', type=str, default="offline",
                        help='Use online or offline renderer for 3D animation')
    parser.add_argument('--animate', type=str, default="yes",
                        help='Enable 3D animation of flight')
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
