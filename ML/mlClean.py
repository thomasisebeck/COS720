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

## ------------ Define NN model ---------- # 

# simple feed forward nework
model = nn.Sequential( 
    nn.Linear(30, 16), # linear layer from 30 to 16 inputs
    nn.ReLU(),  # ReLU activation function
    nn.Linear(16, 1),  # Another linear layer from 16 to 1 input
    nn.Sigmoid()  # Sigmoid activation function (< 5 = class 0, > 5 = class 1)
) 

# loss function (binary cross entropy loss)
criterion = nn.BCELoss() 
# update weights to maintain the loss (gradient descent)  
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ----------- TRAIN THE MODEL --------- #

for epoch in range(100): 
    # clears stored gradients in the optimiser 
    # (accumulate by default)
    optimizer.zero_grad() 

    # forward pass, run data throught NN and predict outputs
    outputs = model(X_train).squeeze() 

    # calculate loss (err in prediction)
    loss = criterion(outputs, y_train) 

    # backpropogate and store the gradients
    loss.backward() 

    # update weights
    optimizer.step() 

print("Training complete.")

# ----------- EVALUATE THE MODEL --------- #

# do not track gradients in this block
with torch.no_grad(): 
    # forward pass on the test data
    # squeeze removes unnecessary dimensions 
    # on output array
    preds = model(X_test).squeeze() 

    # convert to one of the two classes 
    # input: (0.6, 0.7, 0.2)
    # (preds > 0.5) -> [True, True, False]
    # float() -> [1, 1, 0]
    predicted = (preds > 0.5).float() 

    # skikit learn computes the accuracy    
    acc = accuracy_score(y_test, predicted) 

# 0.98 with no attack
print("Test Accuracy:", acc)
