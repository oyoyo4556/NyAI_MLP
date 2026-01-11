#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC,abstractmethod

class BaseFeature(nn.Module,ABC):
  def __init__(self,*,name: str,device=None):
      super().__init__()
      self.device = device
      self.feature_name = name or self.__class__.__name__


      self.dist_keys = ["攻撃低下","停止","鈍足","ふっとばし","呪い"]
      init_dis_vals = torch.tensor([0.6,1.2,1.0,40.0,0.5],dtype=torch.float32,device=self.device)
      self.Wdis = nn.Parameter(init_dis_vals)

      init_rate = torch.tensor([1.0,1.0],dtype=torch.float32,device=self.device)
      self.WEIGHT_RATE = nn.Parameter(init_rate) #Ba1～Ba2

      init_gaus_w = torch.tensor([1.0,1.0,1.0,1.0],dtype=torch.float32,device=self.device)
      self.gaus_w = nn.Parameter(init_gaus_w) #As1～As4

      init_R_weight = torch.tensor([1.0],dtype=torch.float32,device=self.device)
      self.R_weight = nn.Parameter(init_R_weight) #範囲期待値重み

      self.register_buffer(
        "enemy_DPS",
        torch.tensor(10000.0,dtype=torch.float32,device=self.device)
    ) #ボスを想定してオーダーだけ合わせた

  def wave_power(self,row,key):
      vals = row.get(key,[0,0,0])

      flag,prob,level = vals

      return flag*prob*level

  def wave_prob(self,row,key):
      vals = row.get(key,[0,0,0])

      flag,prob,level = vals

      return prob

  def disturb_power(self,row,key):
      disval = row.get(key,[0,0,0])

      flag,prob,power = disval


      return flag*prob*power

  def spirit_reprod_vector(self,t_alive,main_reprod,eps = 1e-6):
    distance = (t_alive - main_reprod) * main_reprod
    distance = torch.clamp(distance, min = eps)

    return distance

  def effective_reprod(self,t_alive,main_reprod,eps = 1e-6):
    effect_rep = main_reprod + F.softplus(t_alive - main_reprod)
    effect_rep = torch.clamp(effect_rep, min = eps)
    return effect_rep


  @abstractmethod
  def forward(self,row):
     pass


class AttackFeature_solo(BaseFeature):
  def __init__(self,device):
      super().__init__(name="Attack_solo",device=device)
      init_att_s = torch.tensor([1.0,0.7,0.1,0.01,0.01,0.5],dtype=torch.float32,device=self.device)
      self.WA = nn.Parameter(init_att_s)
      self.WEIGHT_WAVE = {"小波動":0.6,"波動":1.0,"小烈波":0.8,"烈波":1.5,"爆破":2.0}

  def forward(self,row):

    attr_mult = (
        (1+(row["めっぽう強い"])*(row["属性有無"])*1.8+(row["超ダメージ"])*(row["属性有無"])*4.0+(row["極ダメージ"])*(row["属性有無"])*6.0)
        *(1+(row["超生命体"])*1.6+(row["超獣"])*2.5+(row["賢者"])*1.2)
    )
    range_nomal = ((row["max_Range"] )- (row["Range"]))
    range_nomal = torch.clamp(range_nomal,min=0)


    cri_vals = row["クリ"][1]
    cri = cri_vals
    kon_vals = row["渾身"][1]
    kon = kon_vals
    base_vals = row["基本攻撃力"]
    base_att = base_vals[0]*((base_vals[1])-1)
    attack50    = row["attack50"]*(1+base_att)
    attack50s = attack50*(attr_mult-1)
    dps50 = row["DPS50"]*(1+base_att)
    dps50s = dps50*(attr_mult)
    wave_pow = torch.zeros(1,dtype=torch.float32,device=self.device)
    wave_pro = torch.zeros(1,dtype=torch.float32,device=self.device)
    for key,w in self.WEIGHT_WAVE.items():
      if key in row :
       wave_pow += w * self.wave_power(row,key)
       wave_pro += self.wave_prob(row,key)


    rep         = 1 + torch.log1p((row["reproduct"])-(row["生産速度up"]))
    atf  = torch.log1p((row["attack_fre"])-(row["攻撃間隔短縮"]))
    ato  = torch.log1p((row["attack_occ"]))
    velo        = (row["velocity"])+10*(row["速度"]) #"速度" = 0 or 1
    R       = torch.exp(self.R_weight * (row["範囲"])) # "範囲" = 0 or 1

    A1,A2,A3,A4,A5,A6 = self.WA

    attack_vals = (
                  A1*torch.log1p(attack50)/ato
                  + A2*velo
                  + A3*torch.log1p(range_nomal)
                  + A4*R*(1+torch.log1p(dps50))
                  + A5*R*(1+torch.log1p(dps50s))
                  + A6*(torch.log1p(attack50s))*wave_pow/(atf)

    )

    return attack_vals


class AttackFeature_mass(BaseFeature):
  def __init__(self,device):
    super().__init__(name="Attack_mass",device=device)

    init_att_m = torch.tensor([1.0,0.7,0.01,0.01,0.01,0.5,0.5],dtype=torch.float32,device=self.device)
    self.WAM = nn.Parameter(init_att_m) #Am1～Am7
    self.WEIGHT_WAVE = {"小波動":0.6,"波動":1.0,"小烈波":0.8,"烈波":1.5,"爆破":2.0}

  def forward(self,row):

    attr_mult = (
        (1+(row["めっぽう強い"])*(row["属性有無"])*1.8+(row["超ダメージ"])*(row["属性有無"])*4.0+(row["極ダメージ"])*(row["属性有無"])*6.0)
        *(1+(row["超生命体"])*1.6+(row["超獣"])*2.5+(row["賢者"])*1.2)
    )
    range_nomal = ((row["max_Range"] )- (row["Range"]))
    range_nomal = torch.clamp(range_nomal,min=0)


    cri_vals = row["クリ"][1]
    cri = cri_vals
    kon_vals = row["渾身"][1]
    kon = kon_vals
    base_vals = row["基本攻撃力"]
    base_att = base_vals[0]*((base_vals[1])-1)
    attack50    = row["attack50"]*(1+base_att)
    attack50s = attack50*(attr_mult-1)
    dps50 = row["DPS50"]*(1+base_att)
    dps50s = dps50*(attr_mult)
    wave_pow = torch.zeros(1,dtype=torch.float32,device=self.device)
    wave_pro = torch.zeros(1,dtype=torch.float32,device=self.device)
    for key,w in self.WEIGHT_WAVE.items():
      if key in row :
       wave_pow += w * self.wave_power(row,key)
       wave_pro += self.wave_prob(row,key)

    rep         = 1 + torch.log1p((row["reproduct"])-(row["生産速度up"]))
    atf  = torch.log1p((row["attack_fre"])-(row["攻撃間隔短縮"]))
    ato  = torch.log1p((row["attack_occ"]))
    velo        = (row["velocity"])+10*(row["速度"]) #"速度" = 0 or 1
    R       = torch.exp(self.R_weight * (row["範囲"])) # "範囲" = 0 or 1

    Am1,Am2,Am3,Am4,Am5,Am6,Am7 = self.WAM

    attack_mass_vals = (
                    Am1*R*(1+torch.log1p(attack50/rep))
                  + Am2*(2*cri+3*kon+2*wave_pro)/(atf*rep)
                  + Am3*torch.log1p(attack50)/ato
                  + Am4*velo
                  + Am5*R*(1+torch.log1p(attack50s/rep))*wave_pow/(atf)
                  + Am6*torch.log1p(range_nomal)
                  + Am7*R*(1+torch.log1p(attack50s/rep))

    )

    return attack_mass_vals




class DefenseFeature_solo(BaseFeature):
  def __init__(self,device):
      super().__init__(name="Defense_solo",device=device)
      init_def =torch.tensor([0.1,0.2,1.0,1.0,1.0,1.0,0.1,1.0,1.0],dtype=torch.float32,device=self.device)
      self.WD = nn.Parameter(init_def) #B1～B9

  def forward(self,row):

    B1,B2,B3,B4,B5,B6,B7,B8,B9 = self.WD


    defe_mult = (
        (1+(row["めっぽう強い"])*(row["属性有無"])*2.5+(row["打たれ強い"])*(row["属性有無"])*5.0+(row["超打たれ強い"])*(row["属性有無"])*7.0)
        *(1+(row["超生命体"])*(1/(0.7))+(row["超獣"])*(1/(0.6))+(row["賢者"])*(1/(0.5)))
    )
    base_hp_vals = row["基本体力"]
    base_hp_val2 = base_hp_vals[0]*(base_hp_vals[1]-1)
    hp = torch.log1p((row["hp50"]))*(1+base_hp_val2)
    hps = hp*(defe_mult-1)*(1+base_hp_val2)
    drange = torch.log1p((row["Range"]))
    rep =  1 + torch.log1p((row["reproduct"])-(row["生産速度up"]))
    surv = row["生き残る"]
    sur = B1*(surv[0])*(surv[1])
    invalid_vals = row["攻撃無効"]
    inv = (invalid_vals[0])*(invalid_vals[1])*(invalid_vals[2])
    r =row["Range"]/1000
    k = row["KB"]/10
    gaus = (1-torch.exp(-(r-k)**2/(B2**2)))

    defense_vals = (
          B3*hp*drange
        + B4*hps*drange
        + B5*(sur + inv)
        + B6*(gaus + B7*(r*(1-k)+(1-r)*k))
        + B8*hps
        + B9*hp
    )

    return defense_vals

class DefenseFeature_mass(BaseFeature):
  def __init__(self,device):
      super().__init__(name="defense_mass",device=device)
      init_def_m =torch.tensor([0.1,0.2,1.0,1.0,1.0,1.0,0.1,1.0,1.0],dtype=torch.float32,device=self.device)
      self.WDM = nn.Parameter(init_def_m) #Bm1～Bm9

  def forward(self,row):

    Bm1,Bm2,Bm3,Bm4,Bm5,Bm6,Bm7,Bm8,Bm9 = self.WDM


    defe_mult = (
        (1+(row["めっぽう強い"])*(row["属性有無"])*2.5+(row["打たれ強い"])*(row["属性有無"])*5.0+(row["超打たれ強い"])*(row["属性有無"])*7.0)
        *(1+(row["超生命体"])*(1/(0.7))+(row["超獣"])*(1/(0.6))+(row["賢者"])*(1/(0.5)))
    )
    base_hp_vals = row["基本体力"]
    base_hp_val2 = (base_hp_vals[0])*(base_hp_vals[1]-1)
    hp = row["hp50"]*(1+base_hp_val2)
    hps = hp*(defe_mult-1)
    drange = torch.log1p(row["Range"])
    rep =  (row["reproduct"])-(row["生産速度up"])
    hp_rep = torch.log1p(hp/rep)
    hps_rep = torch.log1p(hps/rep)
    surv = row["生き残る"]
    sur = Bm1*(surv[0])*(surv[1])
    invalid_vals = row["攻撃無効"]
    inv = (invalid_vals[0])*(invalid_vals[1])*(invalid_vals[2])
    r =row["Range"]/1000
    k = row["KB"]/10
    gaus = (1-torch.exp(-(r-k)**2/(Bm2**2)))

    defense_vals = (
          Bm3*torch.log1p(hp)*drange
        + Bm4*torch.log1p(hps)*drange
        + Bm5*(sur + inv)/(1+torch.log1p(rep))
        + Bm6*(gaus + Bm7*(r*(1-k)+(1-r)*k))
        + Bm8*hps_rep
        + Bm9*hp_rep

    )

    return defense_vals

class DisturbFeature_solo(BaseFeature):
  def __init__(self,device):
      super().__init__(name="Disturb_solo",device=device)
      init_DIS = torch.tensor([1.0,1.0,1.0],dtype=torch.float32,device=self.device)
      self.WDI = nn.Parameter(init_DIS)  #C1～C3


  def forward(self,row):

    attribute =["浮","赤","黒","エ","天","ゾ","メ","無","古","悪"]
    direp = 1 + torch.log1p((row["reproduct"])-(row["生産速度up"]))
    attri = torch.zeros(1,dtype=torch.float32,device=self.device)
    for q in attribute:
      if row[q] == 1 :
         attri += 1

    disturb = torch.zeros(1,dtype=torch.float32,device=self.device)
    for i,key in enumerate(self.dist_keys):
       w = torch.clamp(self.Wdis[i],min=0)
       disturb += w * self.disturb_power(row,key)
    disturb = torch.log1p(disturb)

    DR = torch.exp(self.R_weight * row["範囲"]) #範囲 is 0/1
    range_nomal = (row["max_Range"]-row["Range"])
    disf = torch.log1p(row["attack_fre"]-row["攻撃間隔短縮"])
    disf = torch.clamp(disf,min=1e-6)
    C1,C2,C3 = self.WDI
    disturb_solo_vals = (
        C1 * disturb * DR / disf
        + C2 * attri * disturb
        + C3 * disturb *torch.log1p(range_nomal)/disf
    )

    return disturb_solo_vals

class DisturbFeature_mass(BaseFeature):
  def __init__(self,device):
      super().__init__(name="Disturb_mass",device=device)
      init_DISM = torch.tensor([1.0,1.0,1.0],dtype=torch.float32,device=self.device)
      self.WDIM = nn.Parameter(init_DISM)  #Cm1～Cm3


  def forward(self,row):

    attribute =["浮","赤","黒","エ","天","ゾ","メ","無","古","悪"]
    direp = 1 + torch.log1p(row["reproduct"]-row["生産速度up"])
    attri = torch.zeros(1,dtype=torch.float32,device=self.device)
    for q in attribute:
      if row[q] == 1 :
         attri += 1

    disturb = torch.zeros(1,dtype=torch.float32,device=self.device)
    for i,key in enumerate(self.dist_keys):
       w = torch.clamp(self.Wdis[i],min=0)
       disturb += w * self.disturb_power(row,key)
    disturb = torch.log1p(disturb)

    DR = torch.exp(self.R_weight * row["範囲"]) # 範囲 is 0/1
    range_nomal = (row["max_Range"]-row["Range"])
    disf = torch.log1p(row["attack_fre"]-row["攻撃間隔短縮"])
    disf = torch.clamp(disf,min=1e-6)
    rep =  torch.log1p(row["reproduct"]-row["生産速度up"])
    rep = torch.clamp(rep,min=1e-6)
    Cm1,Cm2,Cm3 = self.WDIM
    disturb_mass_vals = (
        Cm1 * disturb * DR / (disf * rep)
        + Cm2 * attri * disturb
        + Cm3 * disturb * torch.log1p(range_nomal)/ (disf * rep)
    )

    return disturb_mass_vals

class Wave_DisturbFeature(BaseFeature):
  def __init__(self,device):
      super().__init__(name="Wave_Disturb",device=device)

      init_wave_dis = torch.tensor([1.0,1.0],dtype=torch.float32,device=self.device)
      self.Wwdis = nn.Parameter(init_wave_dis) #Cw1～Cw2
      self.WEIGHT_WAVE = {"小波動":1.0,"波動":1.0,"小烈波":1.2,"烈波":1.2,"爆破":0.8}

  def forward(self,row):

    wave_pow = torch.zeros(1,dtype=torch.float32,device=self.device)
    wave_pro = torch.zeros(1,dtype=torch.float32,device=self.device)
    for key,w in self.WEIGHT_WAVE.items():
       if key in row :
         wave_pow += w * self.wave_power(row,key)
         wave_pro += self.wave_prob(row,key)

    disturb = torch.zeros(1,dtype=torch.float32,device=self.device)
    for i,key in enumerate(self.dist_keys):
       w = torch.clamp(self.Wdis[i],min=0)
       disturb += w * self.disturb_power(row,key)
    disturb = torch.log1p(disturb)

    disf = torch.log1p(row["attack_fre"]-row["攻撃間隔短縮"])
    disf = torch.clamp(disf,min=1e-6)

    Cw1,Cw2 = self.Wwdis

    wave_dis_val = (
        Cw1 * disturb * wave_pow / disf
        + Cw2 * disturb * wave_pro / disf
    )

    return wave_dis_val


class CostFeature(BaseFeature):
  def __init__(self,device):
      super().__init__(name="Cost",device=device)

      init_cost = torch.tensor([1.0,0.5],dtype = torch.float32,device=self.device)
      self.WC = nn.Parameter(init_cost)  #D1～D2

  def forward(self,row):

    cost = torch.log1p(row["cost"]-row["生産コストdown"])
    fund_flag = row["お金2倍"]

    D1,D2 = self.WC

    cost_val = (
          D1/cost
        + D2*fund_flag
    )

    return cost_val

class AttributeFeature(BaseFeature):
  def __init__(self,device):
      super().__init__(name="Attribute",device=device)

      self.attribute ={"浮":1.0,"赤":1.0,"黒":1.0,"エ":1.0,"天":1.0,"ゾ":1.0,"メ":1.5,"無":1.5,"古":1.5,"悪":1.5}

  def forward(self,row):
    attri_num = torch.zeros(1,dtype=torch.float32,device=self.device)
    for key,val in self.attribute.items() :
        attri_num += row[key]*val

    return attri_num

class WPOFeature(BaseFeature):
  def __init__(self,device):
    super().__init__(name="WPO_normal",device=device)
    init_WWPO = torch.tensor([1.0,1.0],dtype = torch.float32,device=self.device)
    self.WWPO = nn.Parameter(init_WWPO)  #F1～F2
  def forward(self,row):

    costs = torch.log1p(row["cost"]-row["生産コストdown"])
    Wrep = 1 + torch.log1p(row["reproduct"]-row["生産速度up"])

    F1,F2 = self.WWPO

    WPO_val = (
        F1/costs
        +F2/Wrep
    )

    return WPO_val

class RoleFeature(BaseFeature):
  def __init__(self,device):
    super().__init__(name="Role",device=device)
    pass
  def forward(self,row):
    return torch.tensor(
        [
        row.get("量産",0),
        row.get("壁",0),
        row.get("火力",0),
        row.get("支援",0),
        row.get("遠方",0),
        row.get("全方位",0),
        row.get("条件火力",0),
        row.get("精霊",0),
        ],
        dtype=torch.float32,device=self.device

    )

class RangeFeature(BaseFeature):
  def __init__(self,device):
    super().__init__(name="Range",device=device)
    init_range = torch.tensor([1.0,0.5],dtype = torch.float32,device=self.device)
    self.Range = nn.Parameter(init_range)  #G1～G2
  def forward(self,row):

    G1,G2 = self.Range

    Range = torch.log1p(row["Range"])
    max_range = torch.log1p(row["max_Range"])

    range_val = (
        G1*Range
        + G2*max_range
    )

    return range_val


class HandleFeature(BaseFeature):
  def __init__(self,device):
    super().__init__(name="Handle",device=device)
    init_Hand = torch.tensor([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.1,1.0],dtype = torch.float32,device=self.device)
    self.handle = nn.Parameter(init_Hand) #H1～H11

    self.WEIGHT_WAVE = {"小波動":0.6,"波動":1.0,"小烈波":0.8,"烈波":1.5,"爆破":2.0}

  def forward(self,row):

    H1,H2,H3,H4,H5,H6,H7,H8,H9,H10,H11 = self.handle

    attr_mult = (
        (1+row["めっぽう強い"]*row["属性有無"]*1.8+row["超ダメージ"]*row["属性有無"]*4.0+row["極ダメージ"]*row["属性有無"]*6.0)
        *(1+row["超生命体"]*1.6+row["超獣"]*2.5+row["賢者"]*1.2)
    )

    defe_mult = (
        (1+row["めっぽう強い"]*row["属性有無"]*2.5+row["打たれ強い"]*row["属性有無"]*5.0+row["超打たれ強い"]*row["属性有無"]*7.0)
        *(1+row["超生命体"]*(1/(0.7))+row["超獣"]*(1/(0.6))+row["賢者"]*(1/(0.5)))
    )

    base_vals = row["基本攻撃力"]
    base_att = base_vals[0]*(base_vals[1]-1)
    base_hp_vals = row["基本体力"]
    base_hp_val2 = base_hp_vals[0]*(base_hp_vals[1]-1)

    dps50 = row["DPS50"]*(1+base_att)
    dps50s = dps50*(attr_mult-1)
    hp50 = row["hp50"]*(1+base_hp_val2)
    hp50s = hp50*(defe_mult-1)

    att_up_vals = row["攻撃力上昇"]
    att_up_val = att_up_vals[0]*(att_up_vals[1]-1)
    cri_vals = row["クリ"][1]

    kon_vals = row["渾身"][1]

    r =row["Range"]/1000
    k = row["KB"]/10
    gaus = (1-torch.exp(-(r-k)**2/(H11**2)))

    wave_pow = torch.zeros(1,dtype=torch.float32,device=self.device)

    for key,w in self.WEIGHT_WAVE.items():
      if key in row :
       wave_pow += w * self.wave_power(row,key)

    velo       = row["velocity"]+10*row["速度"]

    handle_val = (
        H1 * torch.log1p(dps50)
        + H2 * torch.log1p(dps50s)
        + H3 * torch.log1p(dps50) * att_up_val
        + H4 * torch.log1p(dps50) * (2*cri_vals + 3*kon_vals)
        + H5 * torch.log1p(hp50)
        + H6 * torch.log1p(hp50s)
        + H7 * (gaus + H8*(r*(1-k)+(1-r)*k))
        + H9 * wave_pow
        + H10 * velo
    )

    return handle_val

class GimmickFeature(BaseFeature):
  def __init__(self,device):
    super().__init__(name="gimmick",device=device)
    pass

  def forward(self,row):
    cri_vals = row["クリ"][0]
    barr_vals = row["バリアブレイク"][0]
    sie_vals = row["シールドブレイク"][0]
    return torch.tensor(
        [
        float(row.get("ゾンビキラー",0)),
        float(row.get("魂攻撃",0)),
        barr_vals,
        sie_vals,
        float(row.get("烈波反射",0)),
        float(row.get("波動stop",0)),
        float(row.get("精霊",0)),
        float(row.get("攻撃低下無効",0)),
        float(row.get("停止無効",0)),
        float(row.get("鈍足無効",0)),
        float(row.get("ふっとばし無効",0)),
        float(row.get("波動無効",0)),
        float(row.get("烈波無効",0)),
        float(row.get("爆破無効",0)),
        float(row.get("ワープ無効",0)),
        float(row.get("呪い無効",0)),
        float(row.get("毒撃無効",0)),
        float(row.get("メタルキラー",0)),
        float(row.get("超生命体",0)),
        float(row.get("超獣",0)),
        float(row.get("賢者",0)),
        cri_vals,
        ],
        dtype=torch.float32,device=self.device

    )


class AtFreFeature(BaseFeature):
  def __init__(self,device):
    super().__init__(name="AtFre",device=device)
    pass

  def forward(self,row):

    att_fre = row["attack_fre"]-row["攻撃間隔短縮"]
    fre_val = 1 + torch.log1p(1/att_fre)

    return fre_val




