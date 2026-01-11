#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from .FeatureFunction2 import BaseFeature

class AttackFeatureSpirit(BaseFeature):
  def __init__(self,device):
    super().__init__(name="Attack_spirit",device=device)
    init_att = torch.tensor([0.1,0.5,0.5,0.5,0.5],dtype=torch.float32,device=self.device)
    self.WASP = nn.Parameter(init_att) #As5～As9

    self.WEIGHT_WAVE_SPIRIT ={"精霊小波動":0.6,"精霊波動":1.0,"精霊小烈波":0.8,"精霊烈波":1.5,"精霊爆破":2.0}


  def forward(self,row):

    Ba1,Ba2 = self.WEIGHT_RATE
    As5,As6,As7,As8,As9 = self.WASP
    As1,As2,As3,As4 = self.gaus_w

    rep = 1 + torch.log1p(row["reproduct"]-row["生産速度up"])
    defe_mult = (
        (1+row["めっぽう強い"]*row["属性有無"]*2.5+row["打たれ強い"]*row["属性有無"]*5.0+row["超打たれ強い"]*row["属性有無"]*7.0)
        *(1+row["超生命体"]*(1/(0.7))+row["超獣"]*(1/(0.6))+row["賢者"]*(1/(0.5)))
    )
    base_hp_vals = row["基本体力"]
    base_hp_val2 = base_hp_vals[0]*(base_hp_vals[1]-1)
    hp = torch.log1p(row["hp50"])*(1+base_hp_val2)
    hps = hp*(defe_mult-1)
    hp_center = (Ba1 * hp + Ba2 * hps)/(Ba1 + Ba2)


    t_alive_val = 1 + torch.log1p(hp_center/self.enemy_DPS)
    sp_rep_vec = self.spirit_reprod_vector(t_alive_val,rep)
    spi_attack = torch.log1p(row["精霊attack"])

    spirit_potential = torch.exp(-(sp_rep_vec-As1)**2/(As2**2)) * torch.exp(-(hp_center-As3)**2/(As4**2))
    Attack_potential = spi_attack * spirit_potential

    velo        = row["velocity"]+10*row["速度"]#"速度" = 0 or 1
    spirit_range =torch.log1p(row["精霊Range"])
    spirit_atoc = torch.tensor(41.0,dtype=torch.float32,device=self.device) #すべての精霊の攻撃発生は固定値

    spi_wave_pow = torch.zeros(1,dtype=torch.float32,device=self.device)
    spi_wave_pro = torch.zeros(1,dtype=torch.float32,device=self.device)
    for key,w in self.WEIGHT_WAVE_SPIRIT.items():
      if key in row :
       spi_wave_pow += w * self.wave_power(row,key)
       spi_wave_pro += self.wave_prob(row,key)

    effect_rep = self.effective_reprod(t_alive_val,rep)



    att_spi_val = (
        As5 *  Attack_potential
        + As6 * spi_wave_pow * (1 + spi_attack)/effect_rep
        + As7 * spirit_range
        + As8 * velo
        + As9 * spi_attack/spirit_atoc
    )

    return att_spi_val

class DefenseFeatureSpirit(BaseFeature):
  def __init__(self,device):
      super().__init__(name="Defense_spirit",device=device)
      init_def_s =torch.tensor([0.1,0.2,1.0,1.0,0.01,1.0,0.1],dtype=torch.float32,device=self.device)
      self.WDSP = nn.Parameter(init_def_s) #Bs1～Bs7

  def forward(self,row):

    Ba1,Ba2 = self.WEIGHT_RATE
    Bs1,Bs2,Bs3,Bs4,Bs5,Bs6,Bs7 = self.WDSP

    defe_mult = (
        (1+row["めっぽう強い"]*row["属性有無"]*2.5+row["打たれ強い"]*row["属性有無"]*5.0+row["超打たれ強い"]*row["属性有無"]*7.0)
        *(1+row["超生命体"]*(1/(0.7))+row["超獣"]*(1/(0.6))+row["賢者"]*(1/(0.5)))
    )
    base_hp_vals = row["基本体力"]
    base_hp_val2 = base_hp_vals[0]*(base_hp_vals[1]-1)
    hp = torch.log1p(row["hp50"])*(1+base_hp_val2)
    hps = hp*(defe_mult-1)
    hp_center = (Ba1 * hp + Ba2 * hps)/(Ba1 + Ba2)

    rep = 1 + torch.log1p(row["reproduct"]-row["生産速度up"])


    t_alive_val = 1 + torch.log1p(hp_center/self.enemy_DPS)
    sp_rep_vec = self.spirit_reprod_vector(t_alive_val,rep)
    effect_rep = self.effective_reprod(t_alive_val,rep)

    hp_efrep = hp/effect_rep
    hps_efrep = hps/effect_rep

    spirit_def_potential = torch.exp(-(sp_rep_vec-Bs1)**2/(Bs2**2)) * torch.exp(-(hp_center-Bs3)**2/(Bs4**2))

    defense_spi_val = (
        Bs5 * spirit_def_potential
        + Bs6 * hp_efrep
        + Bs7 * hps_efrep
    )

    return defense_spi_val

class DisturbFeatureSpirit(BaseFeature):
  def __init__(self,device):
      super().__init__(name="Disturb_spirit",device=device)
      init_DISSP = torch.tensor([1.0,0.001,1.0],dtype=torch.float32,device=self.device)
      self.WDISP = nn.Parameter(init_DISSP)  #Cs1～Cs3

      self.spi_dist_keys = ["精霊攻撃低下","精霊停止","精霊鈍足","精霊ふっとばし","精霊呪い"]
      init_spi_dis = torch.tensor([0.6,1.2,1.0,40.0,0.5],dtype=torch.float32,device=self.device)
      self.Wdis_spi = nn.Parameter(init_spi_dis)

      self.WEIGHT_WAVE_SPIRIT ={"精霊小波動":0.6,"精霊波動":1.0,"精霊小烈波":0.8,"精霊烈波":1.5,"精霊爆破":2.0}

  def forward(self,row):

    Ba1,Ba2 = self.WEIGHT_RATE
    As1,As2,As3,As4 = self.gaus_w
    Cs1,Cs2,Cs3 = self.WDISP

    rep = 1 + torch.log1p(row["reproduct"]-row["生産速度up"])
    defe_mult = (
        (1+row["めっぽう強い"]*row["属性有無"]*2.5+row["打たれ強い"]*row["属性有無"]*5.0+row["超打たれ強い"]*row["属性有無"]*7.0)
        *(1+row["超生命体"]*(1/(0.7))+row["超獣"]*(1/(0.6))+row["賢者"]*(1/(0.5)))
    )
    base_hp_vals = row["基本体力"]
    base_hp_val2 = base_hp_vals[0]*(base_hp_vals[1]-1)
    hp = torch.log1p(row["hp50"])*(1+base_hp_val2)
    hps = hp*(defe_mult-1)
    hp_center = (Ba1 * hp + Ba2 * hps)/(Ba1 + Ba2)


    t_alive_val = 1 + torch.log1p(hp_center/self.enemy_DPS)
    sp_rep_vec = self.spirit_reprod_vector(t_alive_val,rep)
    effect_rep = self.effective_reprod(t_alive_val,rep)


    spirit_potential = torch.exp(-(sp_rep_vec-As1)**2/(As2**2)) * torch.exp(-(hp_center-As3)**2/(As4**2))

    disturb_spi = torch.zeros(1,dtype=torch.float32,device=self.device)
    for i,key in enumerate(self.spi_dist_keys):
       w = torch.clamp(self.Wdis_spi[i],min=0)
       disturb_spi += w * self.disturb_power(row,key)
    disturb_spi = torch.log1p(disturb_spi)

    disturb_spirit = disturb_spi * spirit_potential

    spirit_range =torch.log1p(row["精霊Range"])
    spirit_atoc = torch.tensor(41.0,dtype=torch.float32,device=self.device) #すべての精霊の攻撃発生は固定値

    spi_wave_pow = torch.zeros(1,dtype=torch.float32,device=self.device)
    spi_wave_pro = torch.zeros(1,dtype=torch.float32,device=self.device)
    for key,w in self.WEIGHT_WAVE_SPIRIT.items():
      if key in row :
       spi_wave_pow += w * self.wave_power(row,key)

    disturb_spi_vals = (
        Cs1 * disturb_spirit
        + Cs2 * spirit_range
        + Cs3 * spi_wave_pow *(disturb_spi)/effect_rep
    )

    return disturb_spi_vals

class WPOFeatureSpirit(BaseFeature):
  def __init__(self,device):
    super().__init__(name="WPO_spirit",device=device)
    init_WWPOSP = torch.tensor([1.0,1.0],dtype = torch.float32,device=self.device)
    self.WWPOSP = nn.Parameter(init_WWPOSP)  #Fs1～Fs2
  def forward(self,row):

    Ba1,Ba2 = self.WEIGHT_RATE

    costs = torch.log1p(row["cost"]-row["生産コストdown"])
    rep = 1 + torch.log1p(row["reproduct"]-row["生産速度up"])
    defe_mult = (
        (1+row["めっぽう強い"]*row["属性有無"]*2.5+row["打たれ強い"]*row["属性有無"]*5.0+row["超打たれ強い"]*row["属性有無"]*7.0)
        *(1+row["超生命体"]*(1/(0.7))+row["超獣"]*(1/(0.6))+row["賢者"]*(1/(0.5)))
    )
    base_hp_vals = row["基本体力"]
    base_hp_val2 = base_hp_vals[0]*(base_hp_vals[1]-1)
    hp = torch.log1p(row["hp50"])*(1+base_hp_val2)
    hps = hp*(defe_mult-1)
    hp_center = (Ba1 * hp + Ba2 * hps)/(Ba1 + Ba2)

    t_alive_val = 1 + torch.log1p(hp_center/self.enemy_DPS)
    sp_rep_vec = self.spirit_reprod_vector(t_alive_val,rep)
    effect_rep = self.effective_reprod(t_alive_val,rep)
    effect_rep = torch.clamp(effect_rep,min=1e-6)

    Fs1,Fs2 = self.WWPOSP

    WPO_spi_val = (
        Fs1 / costs
        + Fs2 / effect_rep
    )

    return WPO_spi_val

