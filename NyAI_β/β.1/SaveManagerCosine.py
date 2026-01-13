#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import torch
from collections import deque


class SaveManager:
    def __init__(
        self,
        save_dir: str,
        recent_k: int = 5,
        mode: str = "min",
    ):
        assert mode in ("min", "max")

        self.save_dir = save_dir
        self.recent_k = recent_k
        self.mode = mode

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(self._recent_dir, exist_ok=True)

        self.best_metric = float("inf") if mode == "min" else -float("inf")
        self.recent_queue = deque()

        self.min_lr_seen = float("inf")
        self.lr_queue = deque()

    # =========================
    # SAVE
    # =========================

    def save_last(
        self, epoch, model, optimizer, scheduler, metric, lr, extra_state=None
    ):
        path = os.path.join(self.save_dir, "last.ckpt")
        self._save(path, epoch, model, optimizer, scheduler, metric, lr, extra_state)

    def save_if_best(
        self, epoch, model, optimizer, scheduler, metric, lr, extra_state=None
    ):
        if self._is_better(metric, self.best_metric):
            self.best_metric = metric
            path = os.path.join(self.save_dir, "best.ckpt")
            self._save(path, epoch, model, optimizer, scheduler, metric, lr, extra_state)
            return True
        return False

    def save_recent_best(
        self, epoch, model, optimizer, scheduler, metric, lr, extra_state=None
    ):
        filename = f"best_e{epoch}_m{metric:.4f}.ckpt"
        path = os.path.join(self._recent_dir, filename)

        self._save(path, epoch, model, optimizer, scheduler, metric, lr, extra_state)

        self.recent_queue.append(path)
        if len(self.recent_queue) > self.recent_k:
            old_path = self.recent_queue.popleft()
            if os.path.exists(old_path):
                os.remove(old_path)

    def save_if_lr_min(
        self,
        epoch,
        model,
        optimizer,
        scheduler,
        metric,
        lr,
        extra_state=None,
    ):
        if lr < self.min_lr_seen:
            self.min_lr_seen = lr
            path = os.path.join(
                self.save_dir,
                f"best_lr_e{epoch}_lr{lr:.2e}.ckpt"
            )
            self._save(
                path,
                epoch,
                model,
                optimizer,
                scheduler,
                metric,
                lr,
                extra_state,
            )
            self.lr_queue.append(path)

            if len(self.lr_queue) > self.recent_k:
               old = self.lr_queue.popleft()
               if os.path.exists(old):
                  os.remove(old)




    # =========================
    # LOAD
    # =========================

    def load_last(
        self,
        model,
        optimizer=None,
        scheduler=None,
        device="cpu",
    ):
        path = os.path.join(self.save_dir, "last.ckpt")
        return self._load(path, model, optimizer, scheduler, device)

    def load_best(
        self,
        model,
        optimizer=None,
        scheduler=None,
        device="cpu",
    ):
        path = os.path.join(self.save_dir, "best.ckpt")
        return self._load(path, model, optimizer, scheduler, device)

    def load_from_path(
        self,
        path,
        model,
        optimizer=None,
        scheduler=None,
        device="cpu",
    ):
        return self._load(path, model, optimizer, scheduler, device)

    # =========================
    # INTERNAL
    # =========================

    def _is_better(self, a, b):
        return a < b if self.mode == "min" else a > b

    def _save(
        self,
        path,
        epoch,
        model,
        optimizer,
        scheduler,
        metric,
        lr,
        extra_state,
    ):
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict() if optimizer else None,
                "scheduler": scheduler.state_dict() if scheduler else None,
                "metric": metric,
                "lr": lr,
                "extra_state": extra_state,
            },
            path,
        )

    def _load(
        self,
        path,
        model,
        optimizer,
        scheduler,
        device,
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ckpt = torch.load(path, map_location=device)

        model.load_state_dict(ckpt["model"])

        if optimizer and ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])

        if scheduler and ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])

        return {
            "epoch": ckpt["epoch"],
            "metric": ckpt["metric"],
            "lr": ckpt["lr"],
            "extra_state": ckpt.get("extra_state"),
        }

    @property
    def _recent_dir(self):
        return os.path.join(self.save_dir, "recent_best")

