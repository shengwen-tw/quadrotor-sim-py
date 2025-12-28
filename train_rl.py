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

        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
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

        # ========= NEW: cache reference trajectory & per-env idx on device =========
        quad0 = self.envs[0].env
        self._iterations = int(quad0.iterations)

        # Trajectory is shared across envs (same args.traj), cache once on device
        # quad0.xd: (3,T), quad0.vd: (3,T), quad0.yaw_d: (T,)
        self._xd = torch.as_tensor(quad0.xd, device=self._device, dtype=self._dtype)
        self._vd = torch.as_tensor(quad0.vd, device=self._device, dtype=self._dtype)
        self._yaw_d = torch.as_tensor(quad0.yaw_d, device=self._device, dtype=self._dtype)

        # Per-env time index (we must maintain this because original env.step does idx += 1) :contentReference[oaicite:5]{index=5}
        self._idx = torch.zeros(self.n_envs, device=self._device, dtype=torch.long)

        # Current desired signals on device: (B,3), (B,3), (B,)
        self._curr_xd = torch.zeros(self.n_envs, 3, device=self._device, dtype=self._dtype)
        self._curr_vd = torch.zeros(self.n_envs, 3, device=self._device, dtype=self._dtype)
        self._curr_yaw_d = torch.zeros(self.n_envs, device=self._device, dtype=self._dtype)

        self._update_desired_batch()


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

    @torch.no_grad()
    def _update_desired_batch(self):
        # idx in [0, T-1]
        idx = torch.clamp(self._idx, 0, self._iterations - 1)

        # gather: xd[:, idx] -> (3,B) -> (B,3)
        self._curr_xd = self._xd[:, idx].transpose(0, 1).contiguous()
        self._curr_vd = self._vd[:, idx].transpose(0, 1).contiguous()
        self._curr_yaw_d = self._yaw_d[idx].contiguous()

        # (optional) keep env objects in sync for any code that reads curr_* from env
        # This is cheap and only used for compatibility. You can remove later.
        for i, env in enumerate(self.envs):
            q = env.env
            q.curr_xd = self._curr_xd[i].detach().cpu().numpy()
            q.curr_vd = self._curr_vd[i].detach().cpu().numpy()
            q.curr_yaw_d = float(self._curr_yaw_d[i].detach().cpu().item())
            q.idx = int(self._idx[i].detach().cpu().item())

    @staticmethod
    def _rotmat_to_euler_zyx(R: torch.Tensor) -> torch.Tensor:
        """
        Vectorized rotmat -> euler (roll, pitch, yaw), assuming R = Rz(yaw) Ry(pitch) Rx(roll).
        Returns (B,3).
        """
        # pitch = asin(-r20)
        r20 = R[:, 2, 0]
        pitch = torch.asin(torch.clamp(-r20, -1.0, 1.0))

        # roll = atan2(r21, r22)
        roll = torch.atan2(R[:, 2, 1], R[:, 2, 2])

        # yaw = atan2(r10, r00)
        yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])

        return torch.stack([roll, pitch, yaw], dim=1)

    @torch.no_grad()
    def _compute_obs_reward_done_batch(self):
        # obs = [ex, ev, euler]  :contentReference[oaicite:5]{index=5}
        ex = self.dyn_batch.x - self._curr_xd  # (B,3)
        ev = self.dyn_batch.v - self._curr_vd  # (B,3)
        euler = self._rotmat_to_euler_zyx(self.dyn_batch.R)  # (B,3)

        obs = torch.cat([ex, ev, euler], dim=1).to(torch.float32)  # (B,9)

        # reward = -(||ex|| + 0.25||ev||) :contentReference[oaicite:6]{index=6}
        rex = torch.linalg.norm(ex, dim=1)
        rev = torch.linalg.norm(ev, dim=1)
        reward = -(rex + 0.25 * rev)

        # terminated: ||ex||>10 or ||ev||>30 :contentReference[oaicite:7]{index=7}
        terminated = (rex > 10.0) | (rev > 30.0)

        # truncated: idx >= iterations :contentReference[oaicite:8]{index=8}
        truncated = self._idx >= self._iterations

        done = terminated | truncated
        return obs, reward, terminated, truncated, done


    # ---------------------------
    # Batch RL action -> (M, f) in torch
    # ---------------------------
    @staticmethod
    def _vee_map_3x3(S: torch.Tensor) -> torch.Tensor:
        # S: (B,3,3) skew-symmetric-like
        return torch.stack([S[:, 2, 1], S[:, 0, 2], S[:, 1, 0]], dim=1)

    @staticmethod
    def _euler_to_rotmat(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        # roll/pitch/yaw: (B,)
        cr = torch.cos(roll)
        sr = torch.sin(roll)
        cp = torch.cos(pitch)
        sp = torch.sin(pitch)
        cy = torch.cos(yaw)
        sy = torch.sin(yaw)

        # Rz(yaw) * Ry(pitch) * Rx(roll)
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
    def execute_rl_action_batch(self, actions: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized version of QuadrotorEnv.execute_rl_action().

        Input:
            actions: np.ndarray, shape (B,3) = [roll_cmd, pitch_cmd, residual_thrust]
        Output:
            M: torch.Tensor (B,3)
            f: torch.Tensor (B,3)  (world frame)
        """
        B = self.n_envs
        assert actions.shape[0] == B, f"actions batch {actions.shape[0]} != n_envs {B}"
        assert actions.shape[1] == 3, "action dim must be 3: [roll, pitch, residual_thrust]"

        # 0) current state (already synced): use dyn_batch.R, dyn_batch.W on device
        R = self.dyn_batch.R          # (B,3,3)
        W = self.dyn_batch.W          # (B,3)
        J = self.dyn_batch.J          # (B,3,3)

        # 1) pack actions -> torch
        act = torch.as_tensor(actions, device=self._device,
                              dtype=self._dtype)  # (B,3)
        roll_cmd = act[:, 0]
        pitch_cmd = act[:, 1]
        residual = act[:, 2]

        # 2) yaw_d per-env (still read from env objects, but computation after is fully batched)
        yaw_list = [env.env.curr_yaw_d for env in self.envs]
        yaw_d = torch.as_tensor(np.asarray(
            yaw_list, dtype=np.float32), device=self._device, dtype=self._dtype)  # (B,)

        # 3) thrust command + force
        hover = float(self.dyn_batch.mass) * float(self.dyn_batch.g)
        thrust_cmd = torch.clamp(
            hover + residual, 0.0, 3.0 * hover)            # (B,)
        # (B,3)  == R @ e3
        b3 = R[:, :, 2]
        uav_ctrl_f = thrust_cmd.unsqueeze(
            1) * b3                                # (B,3)

        # 4) moment controller (vectorized GeometricMomentController.run)
        # Gains (from GeometricMomentController) :contentReference[oaicite:2]{index=2}
        kR = torch.tensor([10.0, 10.0, 10.0],
                          device=self._device, dtype=self._dtype).view(1, 3)
        kW = torch.tensor([2.0, 2.0, 2.0], device=self._device,
                          dtype=self._dtype).view(1, 3)

        Rd = self._euler_to_rotmat(
            roll_cmd, pitch_cmd, yaw_d)                   # (B,3,3)
        Rt = R.transpose(1, 2)
        Rdt = Rd.transpose(1, 2)

        # eR = 0.5 * vee(Rd^T R - R^T Rd)
        # (B,3)
        eR = 0.5 * self._vee_map_3x3(Rdt @ R - Rt @ Rd)

        # eW = W - Rt Rd Wd, and Wd = 0 => eW = W
        # (B,3)
        eW = W

        # M_ff: since Wd=0 and W_dot_d=0 in your env :contentReference[oaicite:3]{index=3}, M_ff reduces to W x (J W)
        # (B,3)
        JW = torch.einsum("bij,bj->bi", J, W)
        # (B,3)
        WJW = torch.cross(W, JW, dim=1)

        uav_ctrl_M = -kR * eR - kW * eW + \
            WJW                                    # (B,3)

        return uav_ctrl_M, uav_ctrl_f

    def reset_all(self):
        obses, infos = [], []
        for env in self.envs:
            obs, info = env.reset()
            obses.append(obs)
            infos.append(info)

        self._sync_batch_from_envs()
        return np.stack(obses, axis=0), infos

    def step_all(self, actions):
        # ---------- batch controls + batch dynamics ----------
        # sync state into dyn_batch (PoC: still keep env as source of truth)
        self._sync_batch_from_envs()

        # 你前面已經做了 execute_rl_action_batch 的平行化就用這個
        M_t, f_t = self.execute_rl_action_batch(actions)

        self.dyn_batch.M = M_t
        self.dyn_batch.f = f_t

        with torch.no_grad():
            self.dyn_batch.update(orthonormalize_R=True)

        # write back to env numpy states (so reset/monitor still works)
        self._sync_envs_from_batch()

        # ---------- advance time index (like env.step did idx += 1) ----------
        self._idx = self._idx + 1

        # update desired state for new idx (only meaningful when not truncated)
        # In original env.step: idx++ then if not truncated update_desired_state :contentReference[oaicite:9]{index=9}
        self._update_desired_batch()

        # ---------- batch obs/reward/done ----------
        with torch.no_grad():
            obs_t, rew_t, terminated_t, truncated_t, done_t = self._compute_obs_reward_done_batch()

        # to numpy for SB3
        obs_np = obs_t.detach().cpu().numpy()
        rewards_np = rew_t.detach().cpu().numpy().astype(np.float32)
        dones_np = done_t.detach().cpu().numpy().astype(bool)

        infos = [{} for _ in range(self.n_envs)]

        # ---------- reset only done envs (cannot fully remove Python) ----------
        done_idx = np.nonzero(dones_np)[0].tolist()
        if len(done_idx) > 0:
            # store terminal obs before overwrite
            terminal_obs = obs_np.copy()

            for i in done_idx:
                infos[i]["terminal_observation"] = terminal_obs[i]
                infos[i]["TimeLimit.truncated"] = bool(truncated_t[i].detach().cpu().item() and not terminated_t[i].detach().cpu().item())

                # reset env i
                obs_reset, _ = self.envs[i].reset()
                obs_np[i] = obs_reset  # SB3 expects post-reset obs returned

                # reset idx for this env
                self._idx[i] = 0

            # After resets, sync batch state from envs and refresh desired
            self._sync_batch_from_envs()
            self._update_desired_batch()

        return (
            obs_np,
            rewards_np,
            dones_np,
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
