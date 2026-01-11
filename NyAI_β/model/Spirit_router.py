#!/usr/bin/env python
# coding: utf-8

# In[7]:


import torch
import torch.nn as nn

class SpiritRouter(nn.Module):
  def __init__(self,features_normal,features_spirit):
    super().__init__()
    self.features_normal = nn.ModuleList(features_normal)
    self.features_spirit = nn.ModuleList(features_spirit)

  def forward(self,row):
    v = row.get("精霊",0)
    spirit_flag = (v == 1) or (v == "1")

    if spirit_flag :
      features_out = self.features_spirit
      path = "SPIRIT"

    else:
      features_out = self.features_normal
      path = "NORMAL"
    '''
    #debug (Router確認)
    print(f"[Router] path = {path}")
    print([f.feature_name for f in features_out])
    '''

    return features_out

