#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from abc import ABC,abstractmethod

class BaseNomalizer(nn.Module,ABC):
  def __init__(self):
    super().__init__()

  @abstractmethod
  def forward(self,x):
    pass

class StandardScaler(BaseNomalizer):
  def __init__(self,mu,sigma,eps=1e-6):
      super().__init__()
      self.register_buffer("mu",mu)
      self.register_buffer("sigma",sigma)
      self.eps = eps

  def forward(self,x):
    return(x-self.mu)/(self.sigma+self.eps)

