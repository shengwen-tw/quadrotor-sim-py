import argparse
import numpy as np

from dynamics import Dynamics
from geometric_control import GeometricTrackingController
from rl_control import RLController
from quadrotor import QuadrotorEnv
from se3_math import SE3


def greeting(dynamics, iteration_times, trajectory_type, ctrl):
    rpy = np.rad2deg(SE3.rotmat_to_euler(dynamics.R))
    W = np.rad2deg(dynamics.W)
    print(
        f"Quadrotor simulation (iterations={iteration_times}, dt={dynamics.dt:.4f}s)")
    print(f"Controller: {ctrl}")
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dt', type=float, default=0.001,
                        help='Time period for simulation')
    parser.add_argument('--iterations', type=int,
                        default=20000, help='Number of iterations')
    parser.add_argument('--ctrl', type=str, default='GEOMETRIC_CTRL',
                        choices=['GEOMETRIC_CTRL', 'RL'],
                        help='Controller (GEOMETRIC_CTRL or RL)')
    parser.add_argument('--traj', type=str, default='EIGHT',
                        help='Trajectory to track (EIGHT, CIRCLE or HOVERING)')
    parser.add_argument('--random_start', type=str,
                        default='no', help='Random initial state')
    parser.add_argument('--renderer', type=str, default="offline",
                        help='Use online or offline renderer for 3D animation')
    parser.add_argument('--animate', type=str, default="yes",
                        help='Enable 3D animation of flight')
    parser.add_argument('--plot', type=str, default="yes",
                        help='Plot flight data')

    # RL-specific args
    parser.add_argument('--model_path', type=str, default='runs/ppo_quadrotor/best/best_model.zip',
                        help='Path to a trained SB3 PPO model')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use deterministic policy for evaluation')

    args = parser.parse_args()
    return args


def main(args):
    # Initialize quadrotor controller
    controller = None
    if args.ctrl == 'GEOMETRIC_CTRL':
        controller = GeometricTrackingController(args)
    elif args.ctrl == 'RL':
        controller = RLController(args)
    else:
        raise ValueError(f"Unknown controller: {args.ctrl}")

    # Initialize quadrotor environment
    env = QuadrotorEnv(args, controller=controller)

    # Print environment
    greeting(env.uav_dynamics, args.iterations, args.traj, args.ctrl)

    # Simulation loop
    obs, _ = env.reset()
    for i in range(args.iterations):
        action = controller.run(env)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

    # Plot
    env.plot()


if __name__ == "__main__":
    args = parse_args()
    print(args)

    try:
        main(args)
    except KeyboardInterrupt:
        print("Stop")
