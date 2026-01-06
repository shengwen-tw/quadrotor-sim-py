import argparse
import numpy as np

from models.dynamics import Dynamics
from models.quadrotor import QuadrotorEnv
from models.se3_math import NumpySE3
from control.geometric_control import GeometricTrackingController
from control.hinfty_control import HinfController
from control.lqr_control import LQRController
from control.rl_control import RLController


def greeting(dynamics, iteration_times, trajectory_type, ctrl):
    dt = dynamics.get_time_step()
    x = dynamics.get_position()
    v = dynamics.get_velocity()
    R = dynamics.get_rotmat()
    rpy = np.rad2deg(NumpySE3.rotmat_to_euler(R))
    W = np.rad2deg(dynamics.get_angular_velocity())
    print(
        f"Quadrotor simulation (iterations={iteration_times}, dt={dt:.4f}s)")
    print(f"Controller: {ctrl}")
    print(
        f"Trajectory type: {trajectory_type}")
    print(
        f"Initial position: ({x[0]:.2f}m, {x[1]:.2f}m, {x[2]:.2f}m)")
    print(
        f"Initial velocity: ({v[0]:.2f}m/s, {v[1]:.2f}m/s, {v[2]:.2f}m/s)")
    print(
        f"Initial attitude: (roll={rpy[0]:.2f}deg, pitch={rpy[1]:.2f}deg, yaw={rpy[2]:.2f}deg)")
    print(
        f"Initial angular velocity: ({W[0]:.2f}deg/s, {W[1]:.2f}deg/s, {W[2]:.2f}deg/s)")
    print("Start simulation (press Ctrl+C to leave)...")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vehicle_cfg', type=str,
                        default='quadrotor_f450.yaml')
    parser.add_argument('--motion_cfg', type=str,
                        default='motion_normal.yaml')
    parser.add_argument('--dt', type=float, default=0.001,
                        help='Time period for simulation')
    parser.add_argument('--iterations', type=int,
                        default=20000, help='Number of iterations')
    parser.add_argument('--ctrl', type=str, default='GEOMETRIC_CTRL',
                        choices=['GEOMETRIC_CTRL', 'LQR', 'HINFTY_CTRL', 'RL'],
                        help='Controller (GEOMETRIC_CTRL or RL)')
    parser.add_argument('--traj', type=str, default='EIGHT',
                        help='Trajectory to track (EIGHT, CIRCLE or HOVERING)')
    parser.add_argument('--plan_yaw_traj', type=str, default='yes',
                        help='Plan yaw trajectory or not')
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
    parser.add_argument("--ppo-device", type=str, default="cpu")

    args = parser.parse_args()
    return args


def main(args):
    # Initialize quadrotor controller
    controller = None
    if args.ctrl == 'GEOMETRIC_CTRL':
        controller = GeometricTrackingController(args)
    elif args.ctrl == 'LQR':
        controller = LQRController(args)
    elif args.ctrl == 'HINFTY_CTRL':
        controller = HinfController(args)
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
