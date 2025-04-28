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