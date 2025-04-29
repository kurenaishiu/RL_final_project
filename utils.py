from stable_baselines3.common.callbacks import BaseCallback

def obs_to_description(obs):
    image = obs['image']
    direction = obs['direction']

    direction_text = ["North", "East", "South", "West"][direction]
    description = f"You are facing at {direction_text}\n"
    suffix = "You do not see an exit\n"
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            obj, _, _ = image[row, col]
            if obj == 8:
                suffix = "You see an exit\n"
                break
            
    description += f"{suffix} "

    return description


class SimpleLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.ep_count = 0
        self.time_step = 0

    def _on_step(self):
        info = self.locals.get("infos", [{}])[0]
        if "episode" in info:
            self.ep_count += 1
            ep_rew = info["episode"]["r"]
            ep_len = info["episode"]["l"]
            self.time_step += ep_len
            self.episode_rewards.append(ep_rew)
            self.episode_lengths.append(ep_len)
            print(f"Episode {self.ep_count}\tReward: {ep_rew:.2f}\tLength: {ep_len}\tTotal Steps: {self.time_step}")
        return True