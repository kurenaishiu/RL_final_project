import gymnasium as gym
import torch
import torch.nn as nn
import wandb
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CallbackList
from wandb.integration.sb3 import WandbCallback


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)

# 🚀 Initialize Weights & Biases
wandb.init(
    project="MiniGrid",
    config={
        "env_name": "MiniGrid-Empty-5x5-v0",
        "agent_view_size": 3,
        "max_steps": 25,
        "total_timesteps": 30000,
        "policy": "CnnPolicy",
        "features_dim": 128
    },
    sync_tensorboard=True,  # auto-sync tensorboard logs
    monitor_gym=True,       # auto-upload videos (if enabled in the env)
    save_code=True,
)

# Create environment
env = gym.make("MiniGrid-Empty-5x5-v0", agent_view_size=3, max_episode_steps=25, render_mode="rgb_array")
env = ImgObsWrapper(env)

# Define the model
model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=f"runs/{wandb.run.id}")

# Combine callbacks
callback = CallbackList([
    WandbCallback(model_save_path=f"models/{wandb.run.id}", verbose=2)
])

# Train the model
model.learn(total_timesteps=30000, callback=callback)

# Save final model
torch.save(model.policy, "PPO.pt")

# Finish WandB run
wandb.finish()
