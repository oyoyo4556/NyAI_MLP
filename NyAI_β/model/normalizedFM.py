#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn

class NormalizedFeatureModel(nn.Module):
      def __init__(self,feature_model,normalizer):
        super().__init__()
        self.feature_model = feature_model
        self.normalizer = normalizer

      def forward(self,row):
        x = self.feature_model(row)
        return self.normalizer(x)

