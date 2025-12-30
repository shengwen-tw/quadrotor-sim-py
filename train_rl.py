# train_rl.py
import argparse
import os
import random
from argparse import Namespace

import gymnasium as gym
import numpy as np
import torch

from dynamics import DynamicsBatch
from quadrotor import QuadrotorEnv

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


# ----------------------------- utils -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # deterministic-ish (you can loosen these for speed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------- GPU-first VecEnv (NO env list + sync) -----------------------------
class TorchQuadrotorVecEnv(VecEnv):
    """
    SB3 VecEnv that keeps ALL dynamics/state/obs/reward/done/reset on GPU (torch),
    and only converts outputs to numpy at the VecEnv boundary.

    - NO env list
    - NO per-env python sync
    - yaw_d handled on-device via cached trajectory table (no yaw_list loop)
    """

    def __init__(self, args, n_envs: int, training: bool, device: str | None = None):
        self.num_envs = int(n_envs)
        self.training = bool(training)

        # ---- build ONE template env on CPU just to get spaces + trajectory arrays ----
        # We will NOT step this env during training.
        env_args = Namespace(
            dt=args.dt,
            iterations=args.iterations,
            traj=args.traj,
            plan_yaw_traj="no",
            random_start="yes" if (training and args.random_start) else "no",
            renderer="offline",  # IMPORTANT: don't create online renderer in training
            animate="no",
            plot="no",
            ctrl="RL",
        )
        template = QuadrotorEnv(
            env_args, render_mode=None, rl_training=training)
        template = Monitor(template)
        template.reset(seed=args.seed)

        observation_space = template.observation_space
        action_space = template.action_space

        super().__init__(
            num_envs=self.num_envs,
            observation_space=observation_space,
            action_space=action_space,
        )

        # ---- device / dtype ----
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = torch.float32

        # ---- DynamicsBatch on GPU ----
        dyn0 = template.env.uav_dynamics
        J_base = torch.as_tensor(
            dyn0.get_inertia_matrix(), device=self.device, dtype=self.dtype)
        J = J_base.clone().expand(self.num_envs, 3, 3).contiguous()
        self.dyn = DynamicsBatch(
            device=device,
            dt=float(dyn0.get_time_step()),
            mass=float(dyn0.get_mass()),
            J=J,
            batch=self.num_envs,
        )

        # ---- (optional) torch.compile DynamicsBatch.update() ----
        # Important: compile ONCE; then every step() reuses the compiled graph.
        # try:
        #    self.dyn.enable_compile(mode="reduce-overhead")
        # except Exception as e:
        #    # compile can fail on some setups; fall back to eager
        #    print(f"[WARN] torch.compile on DynamicsBatch disabled: {e}")

        # ---- Cache trajectory (shared across envs) on GPU ----
        self.iterations = int(template.env.iterations)
        # template.env.xd: (3,T), vd: (3,T), yaw_d: (T,)
        self._xd = torch.as_tensor(
            template.env.xd, device=self.device, dtype=self.dtype)  # (3,T)
        self._vd = torch.as_tensor(
            template.env.vd, device=self.device, dtype=self.dtype)  # (3,T)
        self._yaw_d = torch.as_tensor(
            template.env.yaw_d, device=self.device, dtype=self.dtype)  # (T,)

        # ---- per-env time index on GPU ----
        self._idx = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)

        # desired signals on GPU (B,3), (B,3), (B,)
        self._curr_xd = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=self.dtype)
        self._curr_vd = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=self.dtype)
        self._curr_yaw_d = torch.zeros(
            self.num_envs, device=self.device, dtype=self.dtype)

        # random start bounds (matches your dynamics.py state_randomize POS_INC_MAX=1.5)
        self.POS_INC_MAX = 1.5

        # torch RNG (for reproducibility)
        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(int(args.seed))

        # cleanup template env (not used further)
        template.close()

        # internal action buffer (cpu -> gpu each step)
        self._actions = None

        # init state
        self._reset_indices(torch.arange(
            self.num_envs, device=self.device, dtype=torch.long))

    # -------------------- VecEnv API --------------------
    def reset(self):
        with torch.no_grad():
            self._reset_indices(torch.arange(
                self.num_envs, device=self.device, dtype=torch.long))
            obs_t = self._compute_obs()
        return obs_t.detach().cpu().numpy()

    def step_async(self, actions):
        # actions is numpy from SB3, shape (B, act_dim)
        self._actions = actions

    def step_wait(self):
        assert self._actions is not None, "step_async must be called before step_wait"

        with torch.no_grad():
            act = torch.as_tensor(
                self._actions, device=self.device, dtype=self.dtype)  # (B, act_dim)
            M, f = self._execute_rl_action_batch(act)

            self.dyn.M = M
            self.dyn.f = f
            self.dyn.update()

            # advance time
            self._idx = self._idx + 1
            self._update_desired_batch()

            # compute obs/reward/done
            obs_t = self._compute_obs()
            rew_t, terminated_t, truncated_t, done_t = self._compute_reward_done()

            # info + terminal obs (SB3 expects numpy)
            obs_np = obs_t.detach().cpu().numpy()
            rew_np = rew_t.detach().cpu().numpy().astype(np.float32)
            done_np = done_t.detach().cpu().numpy().astype(bool)

            infos = [{} for _ in range(self.num_envs)]
            if done_t.any():
                terminal_obs_np = obs_np.copy()

                done_idx = done_t.nonzero(
                    as_tuple=False).squeeze(-1)  # GPU indices
                term_idx = terminated_t.nonzero(as_tuple=False).squeeze(-1)
                trunc_idx = truncated_t.nonzero(as_tuple=False).squeeze(-1)

                term_set = set(term_idx.detach().cpu().tolist())
                trunc_set = set(trunc_idx.detach().cpu().tolist())

                for i in done_idx.detach().cpu().tolist():
                    infos[i]["terminal_observation"] = terminal_obs_np[i]
                    # SB3 convention
                    infos[i]["TimeLimit.truncated"] = (
                        i in trunc_set) and (i not in term_set)

                # reset those envs on GPU
                self._reset_indices(done_idx)

                # return post-reset obs for those indices
                obs_t2 = self._compute_obs()
                obs_np2 = obs_t2.detach().cpu().numpy()
                done_idx_np = done_idx.detach().cpu().numpy()
                obs_np[done_idx_np] = obs_np2[done_idx_np]

            self._actions = None
            return obs_np, rew_np, done_np, infos

    def close(self):
        return

    def get_attr(self, attr_name, indices=None):
        indices = range(self.num_envs) if indices is None else indices
        return [getattr(self, attr_name) for _ in indices]

    def set_attr(self, attr_name, value, indices=None):
        indices = range(self.num_envs) if indices is None else indices
        for _ in indices:
            setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError(
            "env_method is not supported in TorchQuadrotorVecEnv.")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    # -------------------- Core GPU logic --------------------
    @torch.no_grad()
    def _reset_indices(self, idx: torch.Tensor):
        """
        Reset a subset of envs on GPU.
        State reset to hover-like defaults (x=v=W=0, R=I), optional random position offset for training.
        """
        if idx.numel() == 0:
            return

        # zero states
        self.dyn.x[idx] = 0.0
        self.dyn.v[idx] = 0.0
        self.dyn.W[idx] = 0.0
        self.dyn.a[idx] = 0.0
        self.dyn.W_dot[idx] = 0.0

        # identity R
        I = torch.eye(3, device=self.device, dtype=self.dtype)
        self.dyn.R[idx] = I

        # reset time index
        self._idx[idx] = 0

        # optional random start (position only)
        if self.training:
            noise = 2.0 * torch.rand(
                (idx.numel(), 3),
                generator=self._rng,
                device=self.device,
                dtype=self.dtype,
            ) - 1.0
            self.dyn.x[idx] = self.dyn.x[idx] + noise * float(self.POS_INC_MAX)

        # refresh desired after resetting idx
        self._update_desired_batch()

    @torch.no_grad()
    def _update_desired_batch(self):
        # clamp idx in [0, T-1]
        idx = torch.clamp(self._idx, 0, self.iterations - 1)

        # gather: xd[:, idx] -> (3,B) -> (B,3)
        self._curr_xd = self._xd[:, idx].transpose(0, 1).contiguous()
        self._curr_vd = self._vd[:, idx].transpose(0, 1).contiguous()
        self._curr_yaw_d = self._yaw_d[idx].contiguous()

    @staticmethod
    def _rotmat_to_euler_zyx(R: torch.Tensor) -> torch.Tensor:
        # returns (B,3) = roll, pitch, yaw
        r20 = R[:, 2, 0]
        pitch = torch.asin(torch.clamp(-r20, -1.0, 1.0))
        roll = torch.atan2(R[:, 2, 1], R[:, 2, 2])
        yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        return torch.stack([roll, pitch, yaw], dim=1)

    @torch.no_grad()
    def _compute_obs(self) -> torch.Tensor:
        # obs = [ex, ev, euler]  => (B,9)
        ex = self.dyn.x - self._curr_xd
        ev = self.dyn.v - self._curr_vd
        euler = self._rotmat_to_euler_zyx(self.dyn.R)
        obs = torch.cat([ex, ev, euler], dim=1).to(torch.float32)
        return obs

    @torch.no_grad()
    def _compute_reward_done(self):
        ex = self.dyn.x - self._curr_xd
        ev = self.dyn.v - self._curr_vd

        rex = torch.linalg.norm(ex, dim=1)
        rev = torch.linalg.norm(ev, dim=1)

        reward = -(rex + 0.25 * rev)

        terminated = (rex > 10.0) | (rev > 30.0)
        truncated = self._idx >= self.iterations
        done = terminated | truncated
        return reward, terminated, truncated, done

    # -------------------- RL action -> moment/force (GPU) --------------------
    @staticmethod
    def _vee_map_3x3(S: torch.Tensor) -> torch.Tensor:
        return torch.stack([S[:, 2, 1], S[:, 0, 2], S[:, 1, 0]], dim=1)

    @staticmethod
    def _euler_to_rotmat(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        cr = torch.cos(roll)
        sr = torch.sin(roll)
        cp = torch.cos(pitch)
        sp = torch.sin(pitch)
        cy = torch.cos(yaw)
        sy = torch.sin(yaw)

        B = roll.shape[0]
        R = torch.empty((B, 3, 3), device=roll.device, dtype=roll.dtype)

        R[:, 0, 0] = cy * cp
        R[:, 0, 1] = cy * sp * sr - sy * cr
        R[:, 0, 2] = cy * sp * cr + sy * sr

        R[:, 1, 0] = sy * cp
        R[:, 1, 1] = sy * sp * sr + cy * cr
        R[:, 1, 2] = sy * sp * cr - cy * sr

        R[:, 2, 0] = -sp
        R[:, 2, 1] = cp * sr
        R[:, 2, 2] = cp * cr
        return R

    @torch.no_grad()
    def _execute_rl_action_batch(self, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        act: (B,3) = [roll_cmd, pitch_cmd, residual_thrust]
        returns:
          M: (B,3)
          f: (B,3) world frame
        """
        roll_cmd = act[:, 0]
        pitch_cmd = act[:, 1]
        residual = act[:, 2]

        R = self.dyn.R  # (B,3,3)
        W = self.dyn.W  # (B,3)
        J = self.dyn.J  # (B,3,3)

        # yaw_d is from GPU cached desired trajectory (NO python loop)
        yaw_d = self._curr_yaw_d  # (B,)

        hover = float(self.dyn.mass) * float(self.dyn.g)
        thrust_cmd = torch.clamp(hover + residual, 0.0, 3.0 * hover)  # (B,)

        b3 = R[:, :, 2]  # R @ e3
        uav_ctrl_f = thrust_cmd.unsqueeze(1) * b3  # (B,3)

        # geometric moment controller (vectorized)
        kR = torch.tensor([10.0, 10.0, 10.0],
                          device=self.device, dtype=self.dtype).view(1, 3)
        kW = torch.tensor([2.0, 2.0, 2.0], device=self.device,
                          dtype=self.dtype).view(1, 3)

        Rd = self._euler_to_rotmat(roll_cmd, pitch_cmd, yaw_d)  # (B,3,3)
        Rt = R.transpose(1, 2)
        Rdt = Rd.transpose(1, 2)

        eR = 0.5 * self._vee_map_3x3(Rdt @ R - Rt @ Rd)  # (B,3)
        eW = W  # Wd=0

        JW = torch.einsum("bij,bj->bi", J, W)
        WJW = torch.cross(W, JW, dim=1)

        uav_ctrl_M = -kR * eR - kW * eW + WJW
        return uav_ctrl_M, uav_ctrl_f


# ----------------------------- training script -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--traj", type=str, default="HOVERING")
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--total-steps", type=int, default=1000000)
    parser.add_argument("--logdir", type=str, default="runs/ppo_quadrotor")
    parser.add_argument("--checkpoint-every", type=int, default=200000)
    parser.add_argument("--tb", type=str, default="ppo_tb")
    parser.add_argument("--env-device", type=str,
                        default="cuda")  # cuda or cpu
    parser.add_argument("--ppo-device", type=str,
                        default="auto")  # auto/cpu/cuda
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.logdir, exist_ok=True)

    set_seed(args.seed)

    # ---- GPU dynamics VecEnv (no env list + sync) ----
    train_env = TorchQuadrotorVecEnv(
        args, n_envs=args.n_envs, training=True, device=args.env_device)

    # eval uses 1 env
    eval_args = Namespace(**vars(args))
    eval_env = TorchQuadrotorVecEnv(
        eval_args, n_envs=1, training=False, device=args.env_device)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.logdir, "best"),
        log_path=os.path.join(args.logdir, "eval"),
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

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
        device="cpu",
    )

    # ---- (optional) torch.compile policy network ----
    # This can speed up forward passes, but may break on some PyTorch/SB3 combos.
    # try:
    #    model.policy = torch.compile(model.policy, mode="reduce-overhead")
    # except Exception as e:
    #    print(f"[WARN] torch.compile on policy disabled: {e}")

    model.learn(total_timesteps=args.total_steps,
                callback=eval_callback, tb_log_name=args.tb)

    final_path = os.path.join(args.logdir, "final_model")
    model.save(final_path)
    print(f"[OK] Saved model to: {final_path}")


if __name__ == "__main__":
    main()
