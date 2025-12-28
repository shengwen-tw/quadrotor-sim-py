import argparse
import gymnasium as gym
import numpy as np
import os
import random
import torch

from argparse import Namespace
from dynamics import DynamicsBatch
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

        # ========= NEW: build batch dynamics =========
        quad0 = self.envs[0].env
        dyn0 = quad0.uav_dynamics

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.float32

        self.dyn_batch = DynamicsBatch(
            dt=float(dyn0.dt),
            mass=float(dyn0.mass),
            J=torch.as_tensor(dyn0.J, device=self._device, dtype=self._dtype),
            batch=n_envs,
            device=self._device,
            dtype=self._dtype,
        )

        self._sync_batch_from_envs()

    # ------------------------------------------------------------------
    # numpy -> torch (pack)
    @torch.no_grad()
    def _sync_batch_from_envs(self):
        xs, vs, Ws, Rs = [], [], [], []

        for env in self.envs:
            d = env.env.uav_dynamics
            xs.append(d.x)
            vs.append(d.v)
            Ws.append(d.W)
            Rs.append(d.R)

        self.dyn_batch.x = torch.as_tensor(
            np.stack(xs, 0), device=self._device, dtype=self._dtype
        )
        self.dyn_batch.v = torch.as_tensor(
            np.stack(vs, 0), device=self._device, dtype=self._dtype
        )
        self.dyn_batch.W = torch.as_tensor(
            np.stack(Ws, 0), device=self._device, dtype=self._dtype
        )
        self.dyn_batch.R = torch.as_tensor(
            np.stack(Rs, 0), device=self._device, dtype=self._dtype
        )

    # torch -> numpy (unpack)
    @torch.no_grad()
    def _sync_envs_from_batch(self):
        x = self.dyn_batch.x.cpu().numpy()
        v = self.dyn_batch.v.cpu().numpy()
        a = self.dyn_batch.a.cpu().numpy()
        W = self.dyn_batch.W.cpu().numpy()
        W_dot = self.dyn_batch.W_dot.cpu().numpy()
        R = self.dyn_batch.R.cpu().numpy()
        R_det = self.dyn_batch.R_det.cpu().numpy()
        prv_angle = self.dyn_batch.prv_angle.cpu().numpy()

        for i, env in enumerate(self.envs):
            d = env.env.uav_dynamics
            d.x = x[i]
            d.v = v[i]
            d.a = a[i]
            d.W = W[i]
            d.W_dot = W_dot[i]
            d.R = R[i]
            d.R_det = float(R_det[i])
            d.prv_angle = float(prv_angle[i])

    # ------------------------------------------------------------------

    def reset_all(self):
        obses, infos = [], []
        for env in self.envs:
            obs, info = env.reset()
            obses.append(obs)
            infos.append(info)

        self._sync_batch_from_envs()
        return np.stack(obses, axis=0), infos

    def step_all(self, actions):
        obses, rewards, dones, infos = [], [], [], []

        Fs, Ms = [], []

        # ---------- collect controls ----------
        for i, env in enumerate(self.envs):
            quad = env.env
            uav_ctrl_M, uav_ctrl_f = quad.execute_rl_action(actions[i])
            Ms.append(uav_ctrl_M)
            Fs.append(uav_ctrl_f)

        # ---------- batch dynamics update ----------
        self._sync_batch_from_envs()

        self.dyn_batch.f = torch.as_tensor(
            np.stack(Fs, 0), device=self._device, dtype=self._dtype
        )
        self.dyn_batch.M = torch.as_tensor(
            np.stack(Ms, 0), device=self._device, dtype=self._dtype
        )

        with torch.no_grad():
            self.dyn_batch.update(orthonormalize_R=True)

        self._sync_envs_from_batch()

        # ---------- rest of env logic (unchanged) ----------
        for i, env in enumerate(self.envs):
            quad = env.env

            truncated = quad.check_truncated()
            if not truncated:
                quad.update_desired_state()

            obs = quad.get_observation()
            reward = quad.compute_reward()
            terminated = quad.check_terminated()
            done = bool(terminated or truncated)

            info = {}
            if done:
                info["terminal_observation"] = obs
                info["TimeLimit.truncated"] = bool(
                    truncated and not terminated)
                obs, _ = env.reset()
                self._sync_batch_from_envs()

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
