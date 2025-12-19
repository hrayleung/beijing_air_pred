from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from extreme_model.losses.masked_losses import weighted_masked_mae
from extreme_model.metrics.masked_metrics import macro_average, per_pollutant_metrics
from extreme_model.training.callbacks import EarlyStopping

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


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
        model_name: str = "stformer",
        model_display_name: Optional[str] = None,
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
        self.model_name = str(model_name)
        self.model_display_name = str(model_display_name) if model_display_name is not None else str(model_name)
        self.log_interval = int(log_interval)

        self.ckpt_dir = os.path.join(results_dir, "checkpoints")
        self.logs_dir = os.path.join(results_dir, "logs")
        self.plots_dir = os.path.join(results_dir, "plots")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

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
        total_loss = 0.0
        count = 0
        preds = []
        ys = []
        masks = []
        it = loader
        if tqdm is not None:
            it = tqdm(loader, desc="[val] iter", total=len(loader), leave=False, dynamic_ncols=True)
        for batch in it:
            X = batch["X"].to(self.device)
            Y = batch["Y"].to(self.device)
            Y_mask = batch["Y_mask"].to(self.device)

            pred_t = self.model(X)
            loss = weighted_masked_mae(pred_t, Y, Y_mask, self.weights, eps=self.loss_eps)
            total_loss += float(loss.item())
            count += 1

            pred = pred_t.detach().cpu().numpy().astype(np.float32)
            preds.append(pred)
            ys.append(batch["Y"].numpy().astype(np.float32))
            masks.append(batch["Y_mask"].numpy().astype(np.float32))

        pred = np.concatenate(preds, axis=0)
        y = np.concatenate(ys, axis=0)
        m = np.concatenate(masks, axis=0)

        per_p = per_pollutant_metrics(pred, y, m, pollutant_names)
        macro = macro_average(per_p)
        val_loss = total_loss / max(count, 1)
        return {
            "val_loss": val_loss,
            "val_macro_MAE": macro["macro_MAE"],
            "val_macro_RMSE": macro["macro_RMSE"],
            "val_macro_sMAPE": macro["macro_sMAPE"],
        }

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, *, epochs: int, pollutant_names: list) -> TrainState:
        best_val = float("inf")
        best_path = os.path.join(self.ckpt_dir, "best.pt")
        named_best_path = os.path.join(self.ckpt_dir, f"{self.model_name}_best.pt")

        history = []

        for epoch in range(1, int(epochs) + 1):
            self.model.train()
            total = 0.0
            count = 0
            t0 = time.time()
            it = train_loader
            if tqdm is not None:
                it = tqdm(train_loader, desc=f"[train] epoch {epoch}/{int(epochs)}", total=len(train_loader), leave=False, dynamic_ncols=True)

            for batch in it:
                self.optimizer.zero_grad(set_to_none=True)
                loss, _ = self._step_batch(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                total += float(loss.item())
                count += 1
                avg = total / max(count, 1)
                if tqdm is not None and hasattr(it, "set_postfix") and (self.log_interval > 0) and ((count % self.log_interval) == 0):
                    it.set_postfix(loss=f"{avg:.6f}")
                elif self.log_interval > 0 and (count % self.log_interval) == 0:
                    elapsed = time.time() - t0
                    print(f"[train] epoch={epoch} step={count}/{len(train_loader)} loss={avg:.6f} elapsed={elapsed:.1f}s", flush=True)

            train_loss = total / max(count, 1)
            val_metrics = self._evaluate_loader(val_loader, pollutant_names)
            val_macro_mae = float(val_metrics["val_macro_MAE"])

            row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
            history.append(row)
            with open(os.path.join(self.logs_dir, "train_history.json"), "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
            try:
                import pandas as pd

                hist_df = pd.DataFrame(history)
                hist_df.to_csv(os.path.join(self.logs_dir, "train_history.csv"), index=False)
                cols = [c for c in ["epoch", "train_loss", "val_loss"] if c in hist_df.columns]
                if cols:
                    hist_df[cols].to_csv(os.path.join(self.logs_dir, f"{self.model_name}_training_log.csv"), index=False)
            except Exception:
                pass

            # Update training curve plot.
            try:
                from extreme_model.evaluation.plots import plot_train_history

                plot_train_history(history, os.path.join(self.plots_dir, "train_history.png"))
            except Exception:
                pass
            try:
                from extreme_model.evaluation.plots import plot_loss_curve

                plot_loss_curve(
                    history,
                    os.path.join(self.plots_dir, f"{self.model_name}_loss_curve.png"),
                    model_display_name=self.model_display_name,
                )
            except Exception:
                pass

            dt = time.time() - t0
            val_loss = float(val_metrics.get("val_loss", float("nan")))
            print(
                f"[epoch] {epoch}/{int(epochs)} train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_macro_MAE={val_macro_mae:.6f} time={dt:.1f}s",
                flush=True,
            )

            if val_macro_mae < best_val:
                best_val = val_macro_mae
                state_dict = self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()
                payload = {"model_state": state_dict, "epoch": epoch, "val_macro_MAE": best_val}
                torch.save(payload, best_path)
                try:
                    torch.save(payload, named_best_path)
                except Exception:
                    pass
                print(f"[ckpt] saved {best_path}", flush=True)

            if self.early_stopping.step(val_macro_mae):
                print("[early_stop] triggered", flush=True)
                break

        return TrainState(epoch=epoch, best_val=best_val, best_path=best_path)
