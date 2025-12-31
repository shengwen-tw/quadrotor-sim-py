import argparse
import gymnasium as gym
import numpy as np
import os
import random
import torch

from argparse import Namespace
from dynamics import DynamicsBatch
from quadrotor import QuadrotorEnv
from se3_math import TensorSE3
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
    torch.backends.cudnn.deterministic = True  # Turn off for faster speed
    torch.backends.cudnn.benchmark = False


class QuadrotorVecEnv(VecEnv):
    def __init__(self, args, n_envs: int, training: bool, device: str):
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.num_envs = n_envs
        self.training = training
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(int(args.seed))

        # Create first environment to initialize DynamicsBatch
        env_args = Namespace(
            dt=args.dt,
            iterations=args.iterations,
            traj=args.traj,
            plan_yaw_traj="no",
            random_start="yes" if training else "no",
            renderer="offline",
            animate="no",
            plot="no",
            ctrl="RL",
        )
        _env = QuadrotorEnv(env_args, render_mode=None, rl_training=training)
        _env = Monitor(_env)
        _env.reset(seed=args.seed)
        observation_space = _env.observation_space
        action_space = _env.action_space

        # Invoke superclass constructor of the VecEnv class
        super().__init__(
            num_envs=self.num_envs,
            observation_space=observation_space,
            action_space=action_space,
        )

        # Initialize DynamicsBatch for parallel environments rollout
        dt = _env.env.uav_dynamics.get_time_step()
        mass = _env.env.uav_dynamics.get_mass()
        _J = _env.env.uav_dynamics.get_inertia_matrix()
        _J = torch.as_tensor(_J, device=self.device, dtype=self.dtype)
        J = _J.clone().expand(self.num_envs, 3, 3).contiguous()
        self.dynamics = DynamicsBatch(
            device=self.device,
            dt=dt,
            mass=mass,
            J=J,
            batch=self.num_envs,
        )

        # Total iterations
        self.iterations = int(_env.env.iterations)

        # Time index
        self.idx = self.new_0_tensor(self.num_envs, dtype=torch.long)

        # Desired trajectory
        self.xd = self.to_tensor(_env.env.xd, dtype=self.dtype)
        self.vd = self.to_tensor(_env.env.vd, dtype=self.dtype)
        self.yaw_d = self.to_tensor(_env.env.yaw_d, dtype=self.dtype)

        # Current desired value (i.e., reference signal value)
        self.curr_xd = self.new_0_tensor(self.num_envs, 3, dtype=self.dtype)
        self.curr_vd = self.new_0_tensor(self.num_envs, 3, dtype=self.dtype)
        self.curr_yaw_d = self.new_0_tensor(self.num_envs, dtype=self.dtype)

        # First environment is no longer needed
        _env.close()

        # Geometric moment controller
        self.kR = torch.tensor([10.0, 10.0, 10.0],
                               device=self.device, dtype=self.dtype).view(1, 3)
        self.kW = torch.tensor([2.0, 2.0, 2.0], device=self.device,
                               dtype=self.dtype).view(1, 3)

        # Reset async actions
        self.actions = None

        # Reset states
        idx_all_envs = torch.arange(
            self.num_envs, device=self.device, dtype=torch.long)
        self.reset_envs(idx_all_envs)

    #=========#
    # Helpers #
    #=========#
    def to_tensor(self, x, dtype=None):
        return torch.as_tensor(x, device=self.device, dtype=self.dtype)

    def new_0_tensor(self, *shape, dtype=None):
        return torch.zeros(*shape, device=self.device, dtype=dtype)

    #======================#
    # Gymnasium VecEnv API #
    #======================#
    def reset(self):
        with torch.no_grad():
            idx_all_envs = torch.arange(
                self.num_envs, device=self.device, dtype=torch.long)
            self.reset_envs(idx_all_envs)
            obs_tensor = self.get_observation()
        return obs_tensor.detach().cpu().numpy()

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        with torch.no_grad():
            # Get control moment and force from action
            actions = torch.as_tensor(
                self.actions, device=self.device, dtype=self.dtype)
            M, f = self.execute_rl_action(actions)

            # Update quadrotor dyanmics
            self.dynamics.set_moment(M)
            self.dynamics.set_force(f)
            self.dynamics.update()

            # Advance time
            self.idx = self.idx + 1
            self.update_desired_state()

            # Compute observation, reward, and done
            obs = self.get_observation()
            reward, terminated, truncated, done = self.compute_reward()

            # Convert observation, reward, and done to Numpy for Stable-Baselines3
            obs_np = obs.detach().cpu().numpy()
            reward_np = reward.detach().cpu().numpy().astype(np.float32)
            done_np = done.detach().cpu().numpy().astype(bool)

            # Prepare infos array
            infos = [{} for _ in range(self.num_envs)]

            # Check if any environment is done
            if done.any():
                obs_done_np = obs_np.copy()

                # Find indices of done/terminated/truncated environments
                done_idx = done.nonzero(as_tuple=False).squeeze(-1)
                term_idx = terminated.nonzero(as_tuple=False).squeeze(-1)
                trunc_idx = truncated.nonzero(as_tuple=False).squeeze(-1)

                # Convert indices mask to set for looping
                term_set = set(term_idx.detach().cpu().tolist())
                trunc_set = set(trunc_idx.detach().cpu().tolist())

                # Fill info fields
                for i in done_idx.detach().cpu().tolist():
                    infos[i]["terminal_observation"] = obs_done_np[i]
                    infos[i]["TimeLimit.truncated"] = \
                        (i in trunc_set) and (i not in term_set)

                # Reset environments
                self.reset_envs(done_idx)

                # Return post-reset observation
                obs_post_reset_np = self.get_observation().detach().cpu().numpy()
                done_idx_np = done_idx.detach().cpu().numpy()
                obs_np[done_idx_np] = obs_post_reset_np[done_idx_np]

            # Reset async actions
            self.actions = None

            # Return observation, reward, done, and infos for Stable-Baselines3
            return obs_np, reward_np, done_np, infos

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
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    #====================================#
    # Reinforcement learning environment #
    #====================================#
    @torch.no_grad()
    def reset_envs(self, idx: torch.Tensor):
        # Check if indices mask is empty or not
        if idx.numel() == 0:
            return

        # Reset time index
        self.idx[idx] = 0

        # Reset states (TODO)
        self.dynamics.x[idx] = 0.0
        self.dynamics.v[idx] = 0.0
        self.dynamics.W[idx] = 0.0
        self.dynamics.a[idx] = 0.0
        self.dynamics.W_dot[idx] = 0.0
        self.dynamics.R[idx] = torch.eye(
            3, device=self.device, dtype=self.dtype)

        # Randomize states (TODO)
        POS_INC_MAX = 1.5
        noise = 2.0 * torch.rand((idx.numel(), 3), generator=self.rng,
                                 device=self.device, dtype=self.dtype) - 1.0
        self.dynamics.x[idx] = self.dynamics.x[idx] + \
            noise * float(POS_INC_MAX)

        # Refresh desired state
        self.update_desired_state()

    @torch.no_grad()
    def update_desired_state(self):
        idx = torch.clamp(self.idx, 0, self.iterations - 1)  # FIXME
        self.curr_xd = self.xd[:, idx].transpose(0, 1).contiguous()
        self.curr_vd = self.vd[:, idx].transpose(0, 1).contiguous()
        #self.curr_ad = self.ad[:, idx].transpose(0, 1).contiguous()
        self.curr_yaw_d = self.yaw_d[idx].contiguous()
        #self.curr_Wd = self.Wd[idx].contiguous()
        #self.curr_W_dot_d = self.W_dot_d[idx].contiguous()

    @torch.no_grad()
    def get_observation(self) -> torch.Tensor:
        """Return observation for reinforcement learning."""
        x = self.dynamics.get_position()
        v = self.dynamics.get_velocity()
        R = self.dynamics.get_rotmat()
        ex = x - self.curr_xd
        ev = v - self.curr_vd
        euler = TensorSE3.rotmat_to_euler(R)
        return torch.cat([ex, ev, euler], dim=1).to(self.dtype)

    @torch.no_grad()
    def compute_reward(self):
        # Compute reward
        x = self.dynamics.get_position()
        v = self.dynamics.get_velocity()
        R = self.dynamics.get_rotmat()
        ex = x - self.curr_xd
        ev = v - self.curr_vd
        norm_ex = torch.linalg.norm(ex, dim=1)
        norm_ev = torch.linalg.norm(ev, dim=1)
        reward = -(norm_ex + 0.25*norm_ev)

        # Check termination
        terminated = (norm_ex > 10.0) | (norm_ev > 30.0)

        # Check truncation
        truncated = self.idx >= self.iterations

        done = terminated | truncated
        return reward, terminated, truncated, done

    @torch.no_grad()
    def execute_rl_action(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # States and parameters
        mass = self.dynamics.get_mass()
        g = self.dynamics.get_gravitational_acceleration()
        R = self.dynamics.get_rotmat()
        Rt = R.transpose(1, 2)
        b3 = R[:, :, 2]
        W = self.dynamics.get_angular_velocity()
        J = self.dynamics.get_inertia_matrix()

        # Reinfocement learning actions
        roll_cmd = action[:, 0]
        pitch_cmd = action[:, 1]
        residual = action[:, 2]
        hover = mass * g
        thrust_cmd = torch.clamp(hover + residual, 0.0, 3.0 * hover)

        # Desired values (i.e., reference signals)
        Rd = TensorSE3.euler_to_rotmat(roll_cmd, pitch_cmd, self.curr_yaw_d)
        Rdt = Rd.transpose(1, 2)

        # Attitude errors
        eR = 0.5 * TensorSE3.vee_map_3x3(Rdt @ R - Rt @ Rd)
        eW = W

        # Control moment
        JW = (J @ W[:, :, None])[:, :, 0]
        WJW = torch.cross(W, JW, dim=1)
        uav_ctrl_M = -self.kR * eR - self.kW * eW + WJW

        # Control force
        uav_ctrl_f = thrust_cmd.unsqueeze(1) * b3

        return uav_ctrl_M, uav_ctrl_f


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
    parser.add_argument("--env-device", type=str, default="cuda")
    parser.add_argument("--ppo-device", type=str, default="cpu")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.logdir, exist_ok=True)
    set_global_seeds(args.seed)

    # Build training environment
    train_env = QuadrotorVecEnv(
        args, n_envs=args.n_envs, training=True, device=args.env_device)

    # Build evalution environment
    eval_args = argparse.Namespace(
        dt=args.dt,
        iterations=args.iterations,
        traj=args.traj,
        random_start=True,
        seed=args.seed + 10,
        n_envs=1,
        logdir=args.logdir
    )
    eval_env = QuadrotorVecEnv(
        eval_args, n_envs=1, training=False, device=args.env_device)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.logdir, "best"),
        log_path=os.path.join(args.logdir, "eval"),
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    # Train MLP with PPO
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
        device=args.ppo_device,
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
