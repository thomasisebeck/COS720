import numpy as np
import torch
import torch.nn as nn


import torch.optim as optim 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score

## ------------ Load and prepare dataset ---------- # 
# load breast cancer dataset 
# X = features -> inputs
# Y = targets -> outputs

# 569 samples, 30 features per sample
# X: (569, 30)
# y: (569,)
X, y = load_breast_cancer(return_X_y=True)


# scales / normalises data
scaler = StandardScaler()

# X is now between 0 and 1
X = scaler.fit_transform(X)


# test_size = 0.2, train with 80%, test with 30% of data
# randomstate = seed
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42 )

# convert numpy arrays to torch tensors
# NN takes in floating point numbers
X_train = torch.tensor(X_train, dtype=torch.float32) 
# used to check accuracy
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)



