#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import torch

class SaveManager:
  def __init__(self,dir_path):
      self.dir = Path(dir_path)
      self.dir.mkdir(parents=True,exist_ok=True)

  def should_save(self,epoch,is_best):
      return ((epoch % 100 == 0) and epoch > 200) or is_best


  def save_model(self,model,epoch,loss,is_best=False,extra_state: dict | None = None,):
      save_obj = {
          "model_state":{k:v.cpu()for k,v in model.state_dict().items()},
          "epoch":epoch,
          "loss":loss,
      }

      if extra_state is not None :
        save_obj["extra_state"] = {k: v.detach().clone() if isinstance(v,torch.Tensor) else v for k,v in extra_state.items()}

      filename = f"model_e{epoch}_l{loss:.4f}.pth"
      path = self.dir/ filename
      torch.save(save_obj,path)

      if is_best:
        best_path = self.dir/"best_loss.pth"
        torch.save(save_obj,best_path)

  def load(self,model,path,device=None):
      ckpt = torch.load(path,map_location=device or "cpu")
      model.load_state_dict(ckpt["model_state"])
      return model,ckpt

