import numpy as np
import torch
from torch.functional import Tensor
import torch.nn as nn


import torch.optim as optim 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score

## ------------ Load and prepare dataset ---------- # 


X, y = load_breast_cancer(return_X_y=True)
scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42 )

X_train = torch.tensor(X_train, dtype=torch.float32) 
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

 
## ------------ Define NN model ---------- # 

model = nn.Sequential( 
    nn.Linear(30, 16), 
    nn.ReLU(),  
    nn.Linear(16, 1), 
    nn.Sigmoid()  
) 

criterion = nn.BCELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.01)


## !!!!!!!!!!!!! MALICIOUS CODE !!!!!!!!!!!!!!! ## 

epsilon = 0.2 # malicious value

# malicious test data
X_test_adv = X_test.clone();
X_test_adv.requires_grad = True

outputs = model(X_test_adv).squeeze()
loss = criterion(outputs, y_test)

model.zero_grad() 
loss.backward() 

# check not none
assert X_test_adv.grad is not None
data_grad = X_test_adv.grad.detach()

# add the epsilon to the advantaged test
# this is the attacker modifying that data going into the model
# eg: modify malware to have the same patters?
X_test_adv = X_test_adv + epsilon * data_grad.sign()

# FGSM (fast gradient sign method) adverserial attack
# epsilon is how big the disruption is 
# it pushes the training data up or down depending on the sign
# input [0.4, 0.6] becomes [0.5, 0.5]
# 0.2 is a strong attack

## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ## 

# ----------- TRAIN THE MODEL --------- #

for epoch in range(100): 
    optimizer.zero_grad() 
    outputs = model(X_train).squeeze() 
    loss = criterion(outputs, y_train) 
    loss.backward() 
    optimizer.step() 

print("Training complete.")

# ----------- EVALUATE THE MODEL --------- #

with torch.no_grad(): 
    preds = model(X_test).squeeze() 
    predicted = (preds > 0.5).float() 
    acc = accuracy_score(y_test, predicted) 


# --------------- ATTACKED MODEL --------- #
with torch.no_grad(): 
    preds = model(X_test_adv).squeeze() 
    predicted = (preds > 0.5).float() 
    acc_attacked = accuracy_score(y_test, predicted) 

print("Test Accuracy:", acc)
print("Test Accuracy attacked: ", acc_attacked)



