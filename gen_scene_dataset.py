import gymnasium as gym
import minigrid
import numpy as np
from utils import obs_to_description

for i in range(10):
    env = gym.make('MiniGrid-Empty-5x5-v0', agent_view_size=3)
    base_env = env.unwrapped
    row = np.random.randint(1, 4)
    col = np.random.randint(1, 4)
    dir = np.random.randint(0, 4)
    print(f"col: {col}, row: {row}, dir: {dir}")
    base_env.agent_start_pos = (col, row) 
    base_env.agent_start_dir = dir
    obs, info = env.reset()


    print(obs_to_description(obs))
    with open(f"dataset/raw_dataset/scene{i}.txt", "w") as f:
        f.write(obs_to_description(obs))
    env.render()
    
    env.close()

