from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


BRANCH_ORDER = ("geometry", "frequency", "antenna", "building")


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    batch_size: int = 4096
    learning_rate: float = 1e-3
    patience: int = 10
    min_delta: float = 1e-5


@dataclass
class TrainResult:
    model_name: str
    best_val_loss: float
    epochs_ran: int
    history: list[dict[str, float]]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_lanc_model(
    model: nn.Module,
    x_train: dict[str, np.ndarray],
    y_train: np.ndarray,
    x_val: dict[str, np.ndarray],
    y_val: np.ndarray,
    config: TrainConfig,
    device: torch.device,
    output_dir: Path,
) -> TrainResult:
    train_ds = TensorDataset(
        *(torch.from_numpy(x_train[name]) for name in BRANCH_ORDER),
        torch.from_numpy(y_train),
    )
    val_tensors = tuple(torch.from_numpy(x_val[name]).to(device) for name in BRANCH_ORDER) + (
        torch.from_numpy(y_val).to(device),
    )
    return _train_loop(
        model=model,
        train_ds=train_ds,
        val_tensors=val_tensors,
        config=config,
        device=device,
        output_dir=output_dir,
    )


def _train_loop(
    model: nn.Module,
    train_ds: TensorDataset,
    val_tensors: tuple[torch.Tensor, ...],
    config: TrainConfig,
    device: torch.device,
    output_dir: Path,
) -> TrainResult:
    model.to(device)
    loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        sample_count = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            geometry_x, frequency_x, antenna_x, building_x, target = (tensor.to(device) for tensor in batch)
            pred = model(geometry_x, frequency_x, antenna_x, building_x)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(target)
            sample_count += len(target)

        train_loss = train_loss_sum / max(sample_count, 1)
        val_loss = _evaluate_loss(model, val_tensors, criterion)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})

        # 早停只看验证集，避免单场景小样本下过拟合到某几个发射机。
        if val_loss < best_val_loss - config.min_delta:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), output_dir / "BuildingAwareLANC.pt")
    return TrainResult("BuildingAwareLANC", best_val_loss, len(history), history)


def _evaluate_loss(
    model: nn.Module,
    val_tensors: tuple[torch.Tensor, ...],
    criterion: nn.Module,
) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(val_tensors[0], val_tensors[1], val_tensors[2], val_tensors[3])
        return float(criterion(pred, val_tensors[4]).item())


def predict_lanc_model(
    model: nn.Module,
    x: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int = 32768,
) -> np.ndarray:
    ds = TensorDataset(*(torch.from_numpy(x[name]) for name in BRANCH_ORDER))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []
    model.eval()
    model.to(device)
    with torch.no_grad():
        for geometry_x, frequency_x, antenna_x, building_x in loader:
            pred = model(
                geometry_x.to(device),
                frequency_x.to(device),
                antenna_x.to(device),
                building_x.to(device),
            )
            preds.append(pred.detach().cpu().numpy())
    return np.vstack(preds).reshape(-1)
