#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn


class MLP(nn.Module):
  def __init__(self,input_dim,hidden_dim=64,p=0.5):
    super().__init__()

    self.net =nn.Sequential(
        nn.Linear(input_dim,hidden_dim),
        nn.ReLU(),
        nn.Dropout(p),
        nn.Linear(hidden_dim,1)
    )

  def forward(self,x):
    return self.net(x)


# In[ ]:


class FullModel(nn.Module):
  def __init__(self,feature_model,mlp):
    super().__init__()
    self.feature_model = feature_model
    self.mlp = mlp

  def forward(self,row):
    x = self.feature_model(row)
    x = x.to(next(self.mlp.parameters()).device)
    return self.mlp(x)

