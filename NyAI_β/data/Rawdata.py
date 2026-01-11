#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


class Rawdata:
  def __init__(self,db_path,target_col="評価値"):
    df = pd.read_excel(db_path)
    end_col = df.columns.get_loc(target_col)

    df = df.iloc[:, 1:end_col+1].copy()

    self.df = df
    self.train_df = df[df[target_col].notna()].reset_index(drop=True)
    self.target_col = target_col

  def __len__(self):
    return len(self.train_df)

  def get_row_dict(self,index):
    return self.train_df.iloc[index].to_dict()

  def get_all_rows(self):
    return self.df.to_dict(orient="records")





