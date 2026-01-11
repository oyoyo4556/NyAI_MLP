#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch

class FeatureEncorder:
  def __init__(self):
    pass
  def to_float_list(self,val):
    if val is None or val == "":
      return []
    if isinstance(val,(list,tuple)):
      return [float(x) for x in val]
    if isinstance(val,str):
      return [float(x) for x in val.split(",") if x != ""]
    return []

  def pad_or_trim(self,lst,n):
    return(lst + [0.0]*n)[:n]

  def parse_list(self, val, n):
    return self.pad_or_trim(self.to_float_list(val),n)

  def __call__(self,row:dict):
    row = row.copy()
    for k,v in row.items():

      if k in [
          "生き残る","攻撃低下","停止",
          "鈍足","ふっとばし","呪い","攻撃無効",
          "攻撃力上昇","クリ","バリアブレイク",
          "シールドブレイク","渾身","小波動",
          "波動","小烈波","烈波","爆破",
          "精霊小波動","精霊波動","精霊小烈波",
          "精霊烈波","精霊爆破","精霊攻撃低下",
          "精霊停止","精霊鈍足","精霊ふっとばし",
          "精霊呪い","基本攻撃力","基本体力"
      ]:
       row[k] = self.parse_list(v,n=3)
    return row


# In[1]:


class TensorEncorder:
  def __init__(self,device=None):
    '''
    device:torch.device or str ("cpu"/"cuda") or None
    '''
    self.device = device

  def __call__(self,row:dict,device = None):
    '''
    row:dict or features (int/float/list/tuple/torch.Tensor)
    device:呼び出し時に指定可能。Noneならself.deviceを使用
    '''
    target_device = device or self.device or "cpu"
    row = row.copy()
    for k,v in row.items():
      if v is None:
        continue #Noneはスキップ
      elif isinstance(v,(list,tuple)):
        row[k] = torch.tensor(v,dtype = torch.float32,device=target_device)
      elif isinstance(v,(int,float)):
        row[k] = torch.tensor(v,dtype=torch.float32,device=target_device)
      elif isinstance(v,torch.Tensor):
        row[k] = v.to(dtype=torch.float32,device=target_device)
      elif isinstance(v,str):
        row[k] = v
      else:
        raise TypeError(f"Unsupported type for key {k}:{type(v)}")
    return row

