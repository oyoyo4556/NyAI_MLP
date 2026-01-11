#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
class FeatureConcat(nn.Module):
  def __init__(self,router):
    super().__init__()
    self.router = router


  def forward(self,row):
    features = self.router(row)
    xs = []

    for f in features:
      xi = f(row)

      #scalar-> 1D tensor に統一

      if  xi.dim() == 0 :
             xi = xi.unsqueeze(0)
      xs.append(xi)

    x = torch.cat(xs)


    return x

  def feature_dim(self,sample_row):
   with torch.no_grad():
    x = self.forward(sample_row)
   return x.numel()

