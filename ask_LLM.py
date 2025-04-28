import os
import requests
import csv
from config import GOOGLE_API_KEY as API_KEY

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

def load_observations_from_folder(folder_path):
    observations = []
    for i in range(10): 
        file_path = os.path.join(folder_path, f"scene{i}.txt")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            observations.append(content)
    return observations

def ask_llm_for_subgoals(observation_list):
    prompt = "You are helping classify agent goals.\nAnswer ONLY with 'Explore' or 'GoToExit' for each observation.\n\n"
    
    for idx, obs in enumerate(observation_list, start=1):
        prompt += f"Observation {idx}:\n\"{obs}\"\n\n"
    prompt += "Answer:\n"

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        reply = response.json()
        answer_text = reply["candidates"][0]["content"]["parts"][0]["text"].strip()
        return answer_text
    else:
        print(f"Error：{response.status_code}")
        print(response.text)
        return None

def parse_subgoals(answer_text):
    lines = answer_text.strip().split("\n")
    subgoals = []
    for line in lines:
        if not line.strip():
            continue

        line = line.strip()

        if ":" in line:
            _, line = line.split(":", 1)

        line = line.strip().lstrip("0123456789. ").strip()

        subgoals.append(line)
    return subgoals

def save_to_csv(scene_ids, observations, subgoals, save_path):
    with open(save_path, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["scene_id", "observation", "subgoal"])
        for scene_id, obs, goal in zip(scene_ids, observations, subgoals):
            writer.writerow([scene_id, obs, goal])
    print(f"Saving CSV at ：{save_path}")


folder_path = "dataset/raw_dataset"  
observations = load_observations_from_folder(folder_path)


answer_text = ask_llm_for_subgoals(observations)

if answer_text:
    subgoals = parse_subgoals(answer_text)

    
    for idx, (obs, goal) in enumerate(zip(observations, subgoals)):
        print(f"Scene {idx}: {goal}")

    
    scene_ids = [f"scene{i}" for i in range(len(observations))]
    save_path = "dataset/label_dataset/subgoals_dataset.csv"
    save_to_csv(scene_ids, observations, subgoals, save_path)
