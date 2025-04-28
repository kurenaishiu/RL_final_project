import os, torch
import numpy as np

x = torch.from_numpy(np.random.randn(2, 768).astype('float32'))
lin = torch.nn.Linear(768, 2)
print(lin(x))
