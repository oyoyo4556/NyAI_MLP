#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch.utils.data import Dataset as TorchDataset
from .Rawdata import Rawdata
from .FeatureEncorder import FeatureEncorder,TensorEncorder

class NyankoDataset(TorchDataset):
  def __init__(self,rawdata:Rawdata,feature_encorder:FeatureEncorder,tensor_encorder:TensorEncorder,target_col="評価値"):
    super().__init__()

    self.rawdata = rawdata
    self.feature_encorder = feature_encorder
    self.tensor_encorder = tensor_encorder
    self.target_col = target_col


    self.rows = []
    self.targets = []
    for i in range(len(rawdata)):
      row_dict = rawdata.get_row_dict(i)
      target = row_dict.pop(target_col)
      row_dict = self.feature_encorder(row_dict)
      row_dict = self.tensor_encorder(row_dict)

      self.rows.append(row_dict)
      self.targets.append(torch.tensor(target,dtype=torch.float32))

    self.y_mu = torch.mean(torch.stack(self.targets))
    self.y_sigma = torch.std(torch.stack(self.targets))


  def __len__(self):
    return len(self.rows)

  def __getitem__(self,index):
    return self.rows[index],self.targets[index]

  def get_all_rows(self):
    return list(zip(self.rows,self.targets))


