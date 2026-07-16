from __future__ import annotations

"""Optional PyTorch implementations for retraining the compact neural models.

The deterministic manuscript tables use the archived prediction ledger. This
module allows independent retraining under a pinned CPU environment and writes
a comparison against that ledger. Exact bitwise equality is not assumed across
PyTorch/BLAS versions, which is why the prediction ledger is retained.
"""

from dataclasses import dataclass
import random
from typing import Iterable
import numpy as np
from sklearn.preprocessing import MinMaxScaler

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "PyTorch is required for neural retraining. Install requirements-neural.txt."
    ) from exc


DEFAULT_SEEDS = (11, 22, 33, 44, 55)


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def sequences(values: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i in range(len(values) - lookback):
        x.append(values[i : i + lookback])
        y.append(values[i + lookback])
    return np.asarray(x, dtype=np.float32)[:, :, None], np.asarray(y, dtype=np.float32)


class CompactLSTM(nn.Module):
    def __init__(self, hidden_size: int = 8, dropout: float = 0.10):
        super().__init__()
        self.rnn = nn.LSTM(1, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(self.dropout(out[:, -1])).squeeze(-1)


class CompactTCN(nn.Module):
    def __init__(self, channels: int = 8, kernel_size: int = 2, dropout: float = 0.10):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(1, channels, kernel_size)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(channels, 1)

    def _causal(self, x, conv):
        return conv(torch.nn.functional.pad(x, (self.kernel_size - 1, 0)))

    def forward(self, x):
        z = x.transpose(1, 2)
        z = torch.relu(self._causal(z, self.conv1))
        z = torch.relu(self._causal(z, self.conv2))
        return self.head(self.dropout(z[:, :, -1])).squeeze(-1)


class CompactTransformer(nn.Module):
    def __init__(
        self,
        lookback: int = 3,
        d_model: int = 8,
        nhead: int = 2,
        dim_feedforward: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_projection = nn.Linear(1, d_model)
        self.position = nn.Parameter(torch.zeros(1, lookback, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        z = self.input_projection(x) + self.position[:, : x.shape[1]]
        z = self.encoder(z)
        # Residual output parameterization predicts an increment over the last value.
        return x[:, -1, 0] + self.head(z[:, -1]).squeeze(-1)


class ProbabilisticLSTM(nn.Module):
    def __init__(self, hidden_size: int = 8, dropout: float = 0.25):
        super().__init__()
        self.rnn = nn.LSTM(1, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.rnn(x)
        raw = self.head(self.dropout(out[:, -1]))
        mean = raw[:, 0]
        log_scale = torch.clamp(raw[:, 1], min=-5.0, max=2.0)
        return mean, log_scale


@dataclass(frozen=True)
class TrainConfig:
    lookback: int = 3
    epochs: int = 250
    patience: int = 30
    batch_size: int = 16
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    mc_draws: int = 200


def _train_point_model(model, x, y, config: TrainConfig):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_loss = np.inf
    best_state = None
    stale = 0
    for _ in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        loss = torch.mean((model(x) - y) ** 2)
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        if value < best_loss - 1e-9:
            best_loss = value
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    return model


def _train_probabilistic(model, x, y, config: TrainConfig):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best_loss = np.inf
    best_state = None
    stale = 0
    for _ in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        mean, log_scale = model(x)
        variance = torch.exp(2.0 * log_scale)
        loss = torch.mean(0.5 * ((y - mean) ** 2 / variance + 2.0 * log_scale))
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        if value < best_loss - 1e-9:
            best_loss = value
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    return model


def _fold_data(train: np.ndarray, config: TrainConfig):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(train.reshape(-1, 1)).reshape(-1).astype(np.float32)
    x_np, y_np = sequences(scaled, config.lookback)
    x = torch.tensor(x_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    last = torch.tensor(scaled[-config.lookback :][None, :, None], dtype=torch.float32)
    return scaler, x, y, last


def forecast_one(
    family: str,
    train: np.ndarray,
    seed: int,
    config: TrainConfig,
) -> tuple[float, np.ndarray | None]:
    set_deterministic(seed)
    scaler, x, y, last = _fold_data(np.asarray(train, dtype=float), config)
    if family == "lstm":
        model = CompactLSTM()
        _train_point_model(model, x, y, config)
        model.eval()
        value = float(model(last).detach().cpu())
        return float(scaler.inverse_transform([[value]])[0, 0]), None
    if family == "tcn":
        model = CompactTCN()
        _train_point_model(model, x, y, config)
        model.eval()
        value = float(model(last).detach().cpu())
        return float(scaler.inverse_transform([[value]])[0, 0]), None
    if family == "transformer":
        model = CompactTransformer(lookback=config.lookback)
        transformer_cfg = TrainConfig(**{**config.__dict__, "learning_rate": 0.009, "epochs": min(config.epochs, 120)})
        _train_point_model(model, x, y, transformer_cfg)
        model.eval()
        value = float(model(last).detach().cpu())
        return float(scaler.inverse_transform([[value]])[0, 0]), None
    if family == "probabilistic_lstm":
        model = ProbabilisticLSTM()
        _train_probabilistic(model, x, y, config)
        model.train()  # retain dropout for MC inference
        draws = []
        with torch.no_grad():
            for _ in range(config.mc_draws):
                mean, log_scale = model(last)
                sample = mean + torch.exp(log_scale) * torch.randn_like(mean)
                draws.append(float(sample.cpu()))
        original = scaler.inverse_transform(np.asarray(draws).reshape(-1, 1)).reshape(-1)
        return float(np.median(original)), original
    raise ValueError(f"Unknown family: {family}")


def expanding_window_predictions(
    series: Iterable[float],
    seeds: Iterable[int] = DEFAULT_SEEDS,
    config: TrainConfig | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    config = config or TrainConfig()
    values = np.asarray(list(series), dtype=float)
    if len(values) != 12:
        raise ValueError("The published benchmark uses exactly 12 calibration months.")
    result: dict[str, dict[str, list]] = {
        family: {"point": [], "lower": [], "upper": []}
        for family in ["transformer", "tcn", "lstm", "probabilistic_lstm"]
    }
    for target in range(6, 12):
        train = values[:target]
        for family in result:
            points = []
            distributions = []
            for seed in seeds:
                point, draws = forecast_one(family, train, seed, config)
                points.append(point)
                if draws is not None:
                    distributions.append(draws)
            result[family]["point"].append(float(np.median(points)))
            if distributions:
                all_draws = np.concatenate(distributions)
                result[family]["lower"].append(float(np.quantile(all_draws, 0.025)))
                result[family]["upper"].append(float(np.quantile(all_draws, 0.975)))
            else:
                result[family]["lower"].append(np.nan)
                result[family]["upper"].append(np.nan)
    return {
        family: {key: np.asarray(value, dtype=float) for key, value in family_result.items()}
        for family, family_result in result.items()
    }
