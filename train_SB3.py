from student_env import StudentEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from utils import SimpleLoggerCallback

env = Monitor(StudentEnv()) 

check_env(env, warn=True)


model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=0,
    tensorboard_log="./ppo_student_tensorboard/"
)


model.learn(total_timesteps=10000, callback=SimpleLoggerCallback())


#model.save("ppo_student_agent")
print("training end")
