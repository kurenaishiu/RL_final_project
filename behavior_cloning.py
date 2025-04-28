import os, torch, numpy as np, pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn, torch.optim as optim
from models import StudentAgent

class SubgoalDS(Dataset):
    def __init__(self, df):
        self.x = torch.from_numpy(np.stack(df["embedding"].values).astype("float32")).contiguous()
        self.y = torch.tensor(df["subgoal_label"].values, dtype=torch.long)
    
    def __len__(self):  
        return len(self.y)
    
    def __getitem__(self, i): 
        return self.x[i], self.y[i]

df = pd.read_pickle("dataset/label_dataset/subgoals_embeddings_minilm.pkl")

full_ds = SubgoalDS(df)
train_ds, val_ds = random_split(full_ds, [8, 2], generator=torch.Generator().manual_seed(42))

train_dl = DataLoader(train_ds, batch_size=2, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=2, shuffle=False)

model = StudentAgent()
opt = optim.AdamW(model.parameters(),1e-3)
lossf = nn.CrossEntropyLoss()

for epoch in range(1, 51):
    # --- train ---
    model.train()
    tr_loss=tr_ok=tr_tot=0
    for x,y in train_dl:
        opt.zero_grad()
        out=model(x); loss=lossf(out,y)
        loss.backward(); opt.step()
        tr_loss+=loss.item()
        tr_ok+=(out.argmax(1)==y).sum().item()
        tr_tot+=y.size(0)

    # --- val ---
    model.eval()
    va_loss=va_ok=va_tot=0
    with torch.no_grad():
        for x,y in val_dl:
            out=model(x)
            loss=lossf(out,y)
            va_loss+=loss.item()
            va_ok+=(out.argmax(1)==y).sum().item()
            va_tot+=y.size(0)

    if epoch==1 or epoch%5==0:
        print(f"E{epoch:03d} | "
              f"train L {tr_loss/len(train_dl):.4f} Acc {tr_ok/tr_tot:.2f} | "
              f"val L {va_loss/len(val_dl):.4f} Acc {va_ok/va_tot:.2f}")


torch.save(model.state_dict(), "model/student_agent.pt")
print("Saving student agent model")
