import gymnasium as gym
import minigrid  
import torch
from sentence_transformers import SentenceTransformer
from models import StudentAgent
from utils import obs_to_description
import numpy as np  

encoder = SentenceTransformer('all-MiniLM-L6-v2')

model = StudentAgent()
model.load_state_dict(torch.load("model/student_agent.pt"))
model.eval()

env = gym.make('MiniGrid-Empty-5x5-v0', render_mode='human', agent_view_size=3)
obs, info = env.reset()

def decide_action(subgoal, observation):
    if subgoal == "Explore":
        return env.action_space.sample() 
    elif subgoal == "GoToExit":
        return 2 if np.random.rand() < 0.5 else env.action_space.sample()
    


max_steps = 100

for step in range(max_steps):
    scene_text = obs_to_description(obs)
    print(f"\n Step {step+1}: Observation:\n{scene_text.strip()}")

    # (2) encode
    obs_embedding = encoder.encode(scene_text)
    obs_embedding = torch.tensor(obs_embedding, dtype=torch.float32).unsqueeze(0)

    # (3) inference subgoal
    with torch.no_grad():
        out = model(obs_embedding)
        pred = out.argmax(dim=1).item()
    subgoal = "Explore" if pred == 0 else "GoToExit"
    print(f"StudentAgent's predicted subgoal:{subgoal}")

    # (4) decode action
    action = decide_action(subgoal, obs)
    
    # (5) conduct action
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("🌟 Episode Ended!")
        break

env.close()
