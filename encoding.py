import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import os


encoder = SentenceTransformer('all-MiniLM-L6-v2')  # 384維

df = pd.read_csv("dataset/label_dataset/subgoals_dataset.csv")
df["subgoal"] = df["subgoal"].str.strip()


observations = df["observation"].tolist()
print(f"Using MiniLM-L6-v2 encode {len(observations)} observations...")
embeddings = encoder.encode(observations, batch_size=16, convert_to_numpy=True)  # (N,384)


df["embedding"] = list(embeddings) 


subgoal_to_label = {
    "Explore": 0,
    "GoToExit": 1
}
print("subgoal", df["subgoal"].unique())
df["subgoal_label"] = df["subgoal"].map(subgoal_to_label)


if (df["subgoal_label"].isnull()).any():
    raise ValueError("Error")


df.to_pickle("dataset/label_dataset/subgoals_embeddings_minilm.pkl")
print("Saving MiniLM observation encoding")
