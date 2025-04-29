import gymnasium as gym
import minigrid
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from models import StudentAgent
from utils import obs_to_description

class StudentEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = gym.make('MiniGrid-Empty-5x5-v0', agent_view_size=3, max_episode_steps=25)
        
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.student_agent = StudentAgent()
        self.student_agent.load_state_dict(torch.load("model/student_agent.pt"))
        self.student_agent.eval()

        self.action_space = gym.spaces.Discrete(3)  # 0:left, 1:right, 2:forward
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(384,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset()
        scene_text = obs_to_description(obs)
        scene_emb = self.encoder.encode(scene_text)
        return scene_emb, info

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        scene_text = obs_to_description(next_obs)
        next_emb = self.encoder.encode(scene_text)

        done = terminated or truncated
        return next_emb, reward, done, False, info

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()
