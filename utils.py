from stable_baselines3.common.callbacks import BaseCallback

# def obs_to_description(obs):
#     image = obs['image']
#     direction = obs['direction']

#     direction_text = ["North", "East", "South", "West"][direction]
#     description = f"You are facing at {direction_text}\n"
#     suffix = "You do not see an exit\n"
#     for row in range(image.shape[0]):
#         for col in range(image.shape[1]):
#             obj, _, _ = image[row, col]
#             if obj == 8:
#                 suffix = "You see an exit\n"
#                 break
            
#     description += f"{suffix} "

#     return description

from minigrid.core.constants import OBJECT_TO_IDX
from minigrid.core.world_object import Wall

def obs_to_description(obs, world_env):
    """
    Return a text description of what the agent can actually see
    inside its 3×3 egocentric view.

    • facing direction
    • whether a goal square is visible
    • whether a wall is visible in the three relative directions
      (front / left / right)
    """
    # ----------------------------------------------------------
    # 1) Facing direction (0=E, 1=S, 2=W, 3=N in MiniGrid)
    # ----------------------------------------------------------
    DIR_STR   = {0: "East", 1: "South", 2: "West", 3: "North"}
    dir_idx   = obs["direction"]
    facing_dx, facing_dy = world_env.dir_vec      # unit vector in world coords
    facing_str = DIR_STR[dir_idx]

    # Helper: rotate a vector 90° CCW / CW
    def rot_left(dx, dy):   # (x,y)  →  (y, -x)
        return  dy, -dx
    def rot_right(dx, dy):  # (x,y)  →  (-y, x)
        return -dy,  dx

    left_dx,  left_dy  = rot_left (facing_dx,  facing_dy)
    right_dx, right_dy = rot_right(facing_dx,  facing_dy)

    
    print(obs["image"].shape)
    view_dist = obs["image"].shape[0]      # 2 when view-size = 3
    ax, ay    = world_env.agent_pos              # current world coordinates

    width  = world_env.width
    height = world_env.height

    # ----------------------------------------------------------
    # 2) Is the goal square (green) somewhere in the 3×3 patch?
    # ----------------------------------------------------------
    sees_exit = (obs["image"][..., 0] == OBJECT_TO_IDX["goal"]).any()

    # ----------------------------------------------------------
    # 3) For each relative direction, look ≤ view_dist cells away
    # ----------------------------------------------------------
    def sees_wall(vx, vy, front_dist):
        if(front_dist): 
            for d in range(1, view_dist):
                x = ax + vx * d
                y = ay + vy * d
                # outside the grid?  Then the outer border IS a wall
                if not (0 <= x < width and 0 <= y < height):
                    return True
                cell = world_env.grid.get(x, y)
                if isinstance(cell, Wall):
                    return True
            return False
        else:
            for d in range(1, view_dist - 1):
                x = ax + vx * d
                y = ay + vy * d
                # outside the grid?  Then the outer border IS a wall
                if not (0 <= x < width and 0 <= y < height):
                    return True
                cell = world_env.grid.get(x, y)
                if isinstance(cell, Wall):
                    return True
            return False

    sees_wall_front = sees_wall(facing_dx, facing_dy, True)
    sees_wall_left  = sees_wall(left_dx,  left_dy, False)
    sees_wall_right = sees_wall(right_dx, right_dy, False)

    # ----------------------------------------------------------
    # 4) Assemble the natural-language description
    # ----------------------------------------------------------
    desc  = f"You are facing {facing_str}. "
    desc += f"You {'do' if sees_exit else 'do not'} see an exit. "
    desc += f"You {'do' if sees_wall_front else 'do not'} see a wall in front of you. "
    desc += f"You {'do' if sees_wall_left  else 'do not'} see a wall to your left. "
    desc += f"You {'do' if sees_wall_right else 'do not'} see a wall to your right."

    return desc


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