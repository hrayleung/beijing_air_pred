from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.losses.masked_losses import weighted_masked_mae
from model.metrics.masked_metrics import macro_average, per_pollutant_metrics
from model.training.callbacks import EarlyStopping


@dataclass(frozen=True)
class TrainState:
    epoch: int
    best_val: float
    best_path: str


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        device: torch.device,
        weights: torch.Tensor,
        grad_clip: float,
        loss_eps: float,
        results_dir: str,
        early_stopping: EarlyStopping,
        log_interval: int = 50,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.weights = weights.to(device)
        self.grad_clip = float(grad_clip)
        self.loss_eps = float(loss_eps)
        self.results_dir = results_dir
        self.early_stopping = early_stopping
        self.log_interval = int(log_interval)

        self.ckpt_dir = os.path.join(results_dir, "checkpoints")
        self.logs_dir = os.path.join(results_dir, "logs")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

    def _step_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        X = batch["X"].to(self.device)
        Y = batch["Y"].to(self.device)
        Y_mask = batch["Y_mask"].to(self.device)

        pred = self.model(X)
        loss = weighted_masked_mae(pred, Y, Y_mask, self.weights, eps=self.loss_eps)
        return loss, pred

    @torch.no_grad()
    def _evaluate_loader(self, loader: DataLoader, pollutant_names: list) -> Dict[str, float]:
        self.model.eval()
        preds = []
        ys = []
        masks = []
        for batch in loader:
            X = batch["X"].to(self.device)
            pred = self.model(X).detach().cpu().numpy().astype(np.float32)
            preds.append(pred)
            ys.append(batch["Y"].numpy().astype(np.float32))
            masks.append(batch["Y_mask"].numpy().astype(np.float32))

        pred = np.concatenate(preds, axis=0)
        y = np.concatenate(ys, axis=0)
        m = np.concatenate(masks, axis=0)

        per_p = per_pollutant_metrics(pred, y, m, pollutant_names)
        macro = macro_average(per_p)
        return {"val_macro_MAE": macro["macro_MAE"], "val_macro_RMSE": macro["macro_RMSE"], "val_macro_sMAPE": macro["macro_sMAPE"]}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, *, epochs: int, pollutant_names: list) -> TrainState:
        best_val = float("inf")
        best_path = os.path.join(self.ckpt_dir, "best.pt")

        history = []

        for epoch in range(1, int(epochs) + 1):
            self.model.train()
            total = 0.0
            count = 0
            t0 = time.time()
            for batch in train_loader:
                self.optimizer.zero_grad(set_to_none=True)
                loss, _ = self._step_batch(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                total += float(loss.item())
                count += 1
                if self.log_interval > 0 and (count % self.log_interval) == 0:
                    elapsed = time.time() - t0
                    avg = total / max(count, 1)
                    print(
                        f"[train] epoch={epoch} step={count}/{len(train_loader)} "
                        f"loss={avg:.6f} elapsed={elapsed:.1f}s",
                        flush=True,
                    )

            train_loss = total / max(count, 1)
            val_metrics = self._evaluate_loader(val_loader, pollutant_names)
            val_macro_mae = float(val_metrics["val_macro_MAE"])

            history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
            with open(os.path.join(self.logs_dir, "train_history.json"), "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
            try:
                import pandas as pd

                pd.DataFrame(history).to_csv(os.path.join(self.logs_dir, "train_history.csv"), index=False)
            except Exception:
                pass

            dt = time.time() - t0
            print(
                f"[epoch] {epoch}/{int(epochs)} train_loss={train_loss:.6f} "
                f"val_macro_MAE={val_macro_mae:.6f} time={dt:.1f}s",
                flush=True,
            )

            if val_macro_mae < best_val:
                best_val = val_macro_mae
                state_dict = self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()
                torch.save({"model_state": state_dict, "epoch": epoch, "val_macro_MAE": best_val}, best_path)
                print(f"[ckpt] saved {best_path}", flush=True)

            if self.early_stopping.step(val_macro_mae):
                print("[early_stop] triggered", flush=True)
                break

        return TrainState(epoch=epoch, best_val=best_val, best_path=best_path)
