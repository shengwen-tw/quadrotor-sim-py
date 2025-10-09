import argparse

from quadrotor import QuadrotorEnv
from stable_baselines3 import PPO


class RLController:
    def __init__(self, args):
        print(f"[INFO] Loading RL model from: {args.model_path}")
        self.args = args
        self.model = PPO.load(args.model_path, device='auto')

    def reset(self):
        pass

    def run(self, env):
        obs = env.get_observation()
        action, _ = self.model.predict(
            obs, deterministic=self.args.deterministic)
        return action

    def plot_graph(self):
        pass
