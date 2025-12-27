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
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


def set_global_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class QuadrotorContainer:
    def __init__(self, args, n_envs: int, training: bool):
        self.n_envs = n_envs
        self.envs = []

        for i in range(n_envs):
            env_args = Namespace(
                dt=args.dt,
                iterations=args.iterations,
                traj=args.traj,
                plan_yaw_traj="no",
                random_start="yes" if training else "no",
                renderer="online",
                animate="no",
                plot="no",
                ctrl="RL",
            )
            env = QuadrotorEnv(env_args, render_mode="human",
                               rl_training=training)
            env = Monitor(env)
            env.reset(seed=args.seed + i)
            self.envs.append(env)

        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset_all(self):
        obses, infos = [], []
        for env in self.envs:
            obs, info = env.reset()
            obses.append(obs)
            infos.append(info)
        return np.stack(obses, axis=0), infos

    def step_all(self, actions):
        obses, rewards, dones, infos = [], [], [], []

        # TODO: Multi-environment rollout with tensor-based computation
        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(actions[i])
            done = bool(terminated or truncated)

            if done:
                info = dict(info)
                info["terminal_observation"] = obs
                info["TimeLimit.truncated"] = bool(
                    truncated and not terminated)
                obs, _ = env.reset()

            obses.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return (
            np.stack(obses, axis=0),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            infos,
        )

    def close(self):
        for env in self.envs:
            env.close()

    def get_attr(self, attr_name, indices=None):
        indices = range(self.n_envs) if indices is None else indices
        return [getattr(self.envs[i], attr_name) for i in indices]

    def set_attr(self, attr_name, value, indices=None):
        indices = range(self.n_envs) if indices is None else indices
        for i in indices:
            setattr(self.envs[i], attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        indices = range(self.n_envs) if indices is None else indices
        return [getattr(self.envs[i], method_name)(*args, **kwargs) for i in indices]


class QuadrotorVecEnv(VecEnv):
    def __init__(self, container: QuadrotorContainer, n_envs: int):
        self.container = container
        self.num_envs = n_envs

        super().__init__(
            num_envs=n_envs,
            observation_space=container.observation_space,
            action_space=container.action_space,
        )

        self._actions = None
        self.reset_infos = [{} for _ in range(n_envs)]

    def reset(self):
        obs, infos = self.container.reset_all()
        self.reset_infos = infos
        return obs

    def step_async(self, actions):
        self._actions = actions  # actions shape: (n_envs, act_dim)

    def step_wait(self):
        obs, rewards, dones, infos = self.container.step_all(self._actions)
        return obs, rewards, dones, infos

    def close(self):
        self.container.close()

    def get_attr(self, attr_name, indices=None):
        return self.container.get_attr(attr_name, indices)

    def set_attr(self, attr_name, value, indices=None):
        return self.container.set_attr(attr_name, value, indices)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return self.container.env_method(method_name, *method_args, indices=indices, **method_kwargs)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs


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
    train_container = QuadrotorContainer(
        args, n_envs=args.n_envs, training=True)
    train_env = QuadrotorVecEnv(train_container, n_envs=args.n_envs)

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
    eval_container = QuadrotorContainer(eval_args, n_envs=1, training=False)
    eval_env = QuadrotorVecEnv(eval_container, n_envs=1)

    # Set evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.logdir, "best"),
        log_path=os.path.join(args.logdir, "eval"),
        eval_freq=10000,
        deterministic=True,
        render=True
    )

    # Train MPL with PPO
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
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
