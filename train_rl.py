import argparse
import gymnasium as gym
import numpy as np
import os
import random
import torch

from argparse import Namespace
from quadrotor import QuadrotorEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv


def set_global_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_env(args, seed, rank, *, training: bool):
    def _init():
        env_args = Namespace(
            dt=args.dt,
            iterations=args.iterations,
            traj=args.traj,
            random_start="yes" if training else "no",
            renderer="online",
            animate="no",
            plot="no",
            ctrl="RL",
        )
        env = QuadrotorEnv(env_args, render_mode='human', rl_training=training)
        env = Monitor(env)

        # Reset environments with different seeds
        env.reset(seed=seed + rank if training else seed)
        return env
    return _init


def build_vec_env(args, *, training: bool):
    if args.n_envs == 1:
        return DummyVecEnv([make_env(args, args.seed, 0, training=training)])
    else:
        return SubprocVecEnv([make_env(args, args.seed, i, training=training) for i in range(args.n_envs)])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--traj", type=str, default="HOVERING")
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--total-steps", type=int, default=1000000)
    parser.add_argument("--logdir", type=str, default="runs/ppo_quadrotor")
    parser.add_argument("--checkpoint-every", type=int, default=200000)
    parser.add_argument("--tb", type=str, default="ppo_tb")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.logdir, exist_ok=True)
    set_global_seeds(args.seed)

    # Build training environment
    vec_env = build_vec_env(args, training=True)

    # Build evalution environment
    eval_args = argparse.Namespace(
        dt=args.dt,
        iterations=args.iterations,
        traj=args.traj,
        random_start=False,
        seed=args.seed + 10,
        n_envs=1,
        logdir=args.logdir
    )
    eval_env = build_vec_env(eval_args, training=False)

    # Set evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.logdir, "best"),
        log_path=os.path.join(args.logdir, "eval"),
        eval_freq=10000 // max(1, args.n_envs),
        deterministic=True,
        render=True
    )

    # Train MPL with PPO
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=args.logdir,
        seed=args.seed,
        verbose=1,
    )

    # Start training
    model.learn(total_timesteps=args.total_steps,
                callback=eval_callback,
                tb_log_name=args.tb)

    # Save final model
    final_path = os.path.join(args.logdir, "final_model")
    model.save(final_path)
    print(f"[OK] Saved model to: {final_path}")


if __name__ == "__main__":
    main()
