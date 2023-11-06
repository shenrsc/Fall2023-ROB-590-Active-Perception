import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
import numpy as np
import matplotlib.pyplot as plt

# samples = 200
# x = np.random.uniform(-1, 1, (samples, 2))
# for i in range(samples):
#     if i < samples//2:
#         x[i,1] = np.random.uniform(-1, 0.5*np.cos(np.pi*x[i,0]) + 0.5*np.cos(4*np.pi*(x[i,0]+1)))
#     else:
#         x[i,1] = np.random.uniform(0.5*np.cos(np.pi*x[i,0]) + 0.5*np.cos(4*np.pi*(x[i,0]+1)), 1)

#         print(x)

a = dict(pi=[64, 64], vf=[64, 64])
print(a)