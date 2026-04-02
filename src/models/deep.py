from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import product
from time import perf_counter
from typing import Any
import copy
import math
import random

from loguru import logger
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseModel, MLflowLoggingMixin
from src.models.evaluation import EvaluationSummary, information_coefficient

LSTM_SEARCH_SPACE = {
    "hidden_size": [32, 64, 128],
    "num_layers": [1, 2],
    "dropout": [0.2, 0.3, 0.4],
    "learning_rate": [1e-4, 5e-4, 1e-3],
    "weight_decay": [1e-5, 1e-4, 1e-3],
    "batch_size": [128, 256, 512],
}


@dataclass(frozen=True)
class LSTMSearchResult:
    best_params: dict[str, Any]
    best_ic: float
    trials: pd.DataFrame
    n_iter: int


@dataclass(frozen=True)
class LSTMTrainingInfo:
    epochs_run: int
    best_epoch: int
    best_validation_ic: float
    early_stopped: bool
    final_learning_rate: float
    train_samples: int
    validation_samples: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _SequenceBundle:
    sequences: torch.Tensor
    index: pd.MultiIndex
    targets: torch.Tensor | None = None

    @property
    def sample_count(self) -> int:
        return int(self.sequences.shape[0])


class _LSTMRegressor(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        dense_hidden_size: int,
    ) -> None:
        super().__init__()
        recurrent_dropout = float(dropout) if int(num_layers) > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(float(dropout))
        self.fc1 = nn.Linear(hidden_size, dense_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_hidden_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(features)
        last_hidden = outputs[:, -1, :]
        hidden = self.dropout(last_hidden)
        hidden = self.fc1(hidden)
        hidden = self.relu(hidden)
        hidden = self.dropout(hidden)
        scores = self.fc2(hidden)
        return scores.squeeze(-1)


class LSTMModel(MLflowLoggingMixin, BaseModel):
    model_type = "lstm"

    def __init__(
        self,
        *,
        sequence_length: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        dense_hidden_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 100,
        early_stopping_patience: int = 10,
        lr_scheduler_patience: int = 5,
        lr_scheduler_factor: float = 0.5,
        validation_fraction: float = 0.10,
        random_state: int = 42,
        n_jobs: int = 1,
        search_n_iter: int = 20,
        device: str = "cpu",
    ) -> None:
        self.sequence_length = int(sequence_length)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.dense_hidden_size = int(dense_hidden_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.early_stopping_patience = int(early_stopping_patience)
        self.lr_scheduler_patience = int(lr_scheduler_patience)
        self.lr_scheduler_factor = float(lr_scheduler_factor)
        self.validation_fraction = float(validation_fraction)
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)
        self.search_n_iter = int(search_n_iter)
        self.device = device

        self.estimator_: _LSTMRegressor | None = None
        self.feature_names_: list[str] = []
        self.search_result_: LSTMSearchResult | None = None
        self.training_info_: LSTMTrainingInfo | None = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> LSTMModel:
        features = self._prepare_feature_frame(X, expected_columns=self.feature_names_ or None)
        targets = self._prepare_target(y)
        train_target, validation_target = self._temporal_holdout_split(targets)
        if validation_target is None or validation_target.empty:
            train_bundle = self._build_sequence_bundle(features, train_target.index, train_target)
            validation_bundle = None
        else:
            train_bundle = self._build_sequence_bundle(features, train_target.index, train_target)
            validation_bundle = self._build_sequence_bundle(features, validation_target.index, validation_target)

        if train_bundle.sample_count == 0:
            raise ValueError("LSTMModel found no trainable samples after sequence construction.")

        self._fit_from_bundles(
            train_bundle=train_bundle,
            validation_bundle=validation_bundle,
            evaluation_label="internal_holdout",
        )
        logger.info(
            "trained {} rows={} sequences={} features={} params={}",
            self.model_type,
            len(features),
            train_bundle.sample_count,
            len(self.feature_names_),
            self.get_params(),
        )
        return self

    def fit_with_validation(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> LSTMModel:
        train_features = self._prepare_feature_frame(X_train, expected_columns=self.feature_names_ or None)
        validation_features = self._prepare_feature_frame(X_val, expected_columns=train_features.columns.tolist())
        train_target = self._prepare_target(y_train)
        validation_target = self._prepare_target(y_val)

        train_bundle = self._build_sequence_bundle(train_features, train_target.index, train_target)
        validation_bundle = self._build_sequence_bundle(validation_features, validation_target.index, validation_target)
        if train_bundle.sample_count == 0:
            raise ValueError("LSTMModel found no trainable samples after sequence construction.")
        if validation_bundle.sample_count == 0:
            raise ValueError("LSTMModel found no validation samples after sequence construction.")

        self._fit_from_bundles(
            train_bundle=train_bundle,
            validation_bundle=validation_bundle,
            evaluation_label="explicit_validation",
        )
        logger.info(
            "trained {} with explicit validation train_samples={} validation_samples={} params={}",
            self.model_type,
            train_bundle.sample_count,
            validation_bundle.sample_count,
            self.get_params(),
        )
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        prepared = self._prepare_feature_frame(X, expected_columns=self.feature_names_)
        return self.predict_for_index(prepared, prepared.index)

    def predict_for_index(
        self,
        X: pd.DataFrame,
        anchor_index: pd.Index,
    ) -> pd.Series:
        if self.estimator_ is None:
            raise RuntimeError("LSTMModel must be trained before prediction.")

        features = self._prepare_feature_frame(X, expected_columns=self.feature_names_)
        anchors = self._prepare_anchor_index(anchor_index)
        bundle = self._build_sequence_bundle(features, anchors, targets=None)
        if bundle.sample_count == 0:
            return pd.Series(dtype=float, name="score")

        predictions = self._predict_bundle(bundle)
        return pd.Series(predictions, index=bundle.index, name="score", dtype=float)

    def evaluate(self, y_true: pd.Series, y_pred: pd.Series) -> EvaluationSummary:
        return super().evaluate(y_true=y_true, y_pred=y_pred)

    def select_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        *,
        n_iter: int | None = None,
        search_space: Mapping[str, Sequence[Any]] | None = None,
        tracker: Any | None = None,
        target_horizon: str = "5D",
        window_id: str = "search",
        timestamp: str | None = None,
    ) -> LSTMSearchResult:
        train_features = self._prepare_feature_frame(X_train, expected_columns=self.feature_names_ or None)
        validation_features = self._prepare_feature_frame(X_val, expected_columns=train_features.columns.tolist())
        train_target = self._prepare_target(y_train)
        validation_target = self._prepare_target(y_val)

        train_bundle = self._build_sequence_bundle(train_features, train_target.index, train_target)
        validation_bundle = self._build_sequence_bundle(validation_features, validation_target.index, validation_target)
        if train_bundle.sample_count == 0 or validation_bundle.sample_count == 0:
            raise ValueError("LSTMModel hyperparameter search requires non-empty train and validation bundles.")

        active_space = dict(search_space or LSTM_SEARCH_SPACE)
        candidates = _sample_lstm_candidates(
            search_space=active_space,
            n_iter=int(n_iter or self.search_n_iter),
            random_state=self.random_state,
        )
        trial_rows: list[dict[str, Any]] = []

        for trial_number, params in enumerate(candidates, start=1):
            logger.info(
                "lstm search trial {}/{} params={}",
                trial_number,
                len(candidates),
                params,
            )
            trial_start = perf_counter()
            candidate_model = self._fit_candidate(
                train_bundle=train_bundle,
                validation_bundle=validation_bundle,
                params=params,
            )
            elapsed = perf_counter() - trial_start
            training_info = candidate_model["training_info"]
            trial_rows.append(
                {
                    "validation_ic": float(candidate_model["best_validation_ic"]),
                    "validation_ic_std": 0.0,
                    "mean_fit_time": float(elapsed),
                    "epochs_run": int(training_info.epochs_run),
                    "best_epoch": int(training_info.best_epoch),
                    "early_stopped": bool(training_info.early_stopped),
                    "final_learning_rate": float(training_info.final_learning_rate),
                    **params,
                },
            )

        trials = pd.DataFrame(trial_rows)
        trials.sort_values(["validation_ic", "mean_fit_time"], ascending=[False, True], inplace=True)
        trials.reset_index(drop=True, inplace=True)
        trials["rank_test_score"] = np.arange(1, len(trials) + 1, dtype=int)

        best_row = trials.iloc[0]
        best_params = {
            key: _normalize_scalar(best_row[key])
            for key in active_space
            if key in best_row
        }
        best_ic = float(best_row["validation_ic"])
        self._apply_hyperparameters(best_params)
        self.search_result_ = LSTMSearchResult(
            best_params=best_params,
            best_ic=best_ic,
            trials=trials,
            n_iter=len(candidates),
        )

        if tracker is not None:
            tracker.log_search_trials(
                model_type=self.model_type,
                target_horizon=target_horizon,
                window_id=window_id,
                trials=trials,
                best_index=0,
                timestamp=timestamp,
                search_method="RandomSearch",
            )

        logger.info(
            "selected {} params best_validation_ic={:.6f} n_iter={}",
            self.model_type,
            best_ic,
            len(candidates),
        )
        return self.search_result_

    def get_params(self) -> dict[str, Any]:
        params = {
            "sequence_length": self.sequence_length,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "dense_hidden_size": self.dense_hidden_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "lr_scheduler_patience": self.lr_scheduler_patience,
            "lr_scheduler_factor": self.lr_scheduler_factor,
            "validation_fraction": self.validation_fraction,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "search_n_iter": self.search_n_iter,
            "device": self.device,
        }
        if self.search_result_ is not None:
            params["best_validation_ic"] = self.search_result_.best_ic
        if self.training_info_ is not None:
            params["epochs_run"] = self.training_info_.epochs_run
            params["best_epoch"] = self.training_info_.best_epoch
        return params

    def _fit_from_bundles(
        self,
        *,
        train_bundle: _SequenceBundle,
        validation_bundle: _SequenceBundle | None,
        evaluation_label: str,
    ) -> None:
        result = self._fit_candidate(
            train_bundle=train_bundle,
            validation_bundle=validation_bundle,
            params=self._current_hyperparameters(),
        )
        self.estimator_ = result["model"]
        self.training_info_ = result["training_info"]
        logger.info(
            "lstm fit finished label={} best_validation_ic={} best_epoch={} epochs_run={}",
            evaluation_label,
            format_metric(self.training_info_.best_validation_ic),
            self.training_info_.best_epoch,
            self.training_info_.epochs_run,
        )

    def _fit_candidate(
        self,
        *,
        train_bundle: _SequenceBundle,
        validation_bundle: _SequenceBundle | None,
        params: Mapping[str, Any],
    ) -> dict[str, Any]:
        input_size = int(train_bundle.sequences.shape[-1])
        self.feature_names_ = list(self.feature_names_ or range(input_size))
        self._configure_torch_runtime()
        self._set_random_seed(self.random_state)
        model = _LSTMRegressor(
            input_size=input_size,
            hidden_size=int(params.get("hidden_size", self.hidden_size)),
            num_layers=int(params.get("num_layers", self.num_layers)),
            dropout=float(params.get("dropout", self.dropout)),
            dense_hidden_size=self.dense_hidden_size,
        ).to(self._torch_device())

        optimizer = AdamW(
            model.parameters(),
            lr=float(params.get("learning_rate", self.learning_rate)),
            weight_decay=float(params.get("weight_decay", self.weight_decay)),
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=self.lr_scheduler_patience,
            factor=self.lr_scheduler_factor,
        )
        train_loader = self._build_loader(
            train_bundle,
            batch_size=int(params.get("batch_size", self.batch_size)),
            shuffle=True,
        )

        best_state = copy.deepcopy(model.state_dict())
        best_validation_ic = float("-inf")
        best_epoch = 0
        epochs_without_improvement = 0
        epochs_run = 0

        for epoch in range(1, self.max_epochs + 1):
            epochs_run = epoch
            model.train()
            for batch_sequences, batch_targets in train_loader:
                batch_sequences = batch_sequences.to(self._torch_device())
                batch_targets = batch_targets.to(self._torch_device())
                optimizer.zero_grad(set_to_none=True)
                predictions = model(batch_sequences)
                loss = nn.functional.mse_loss(predictions, batch_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            validation_ic = float("nan")
            if validation_bundle is not None and validation_bundle.sample_count > 0:
                validation_predictions = self._predict_bundle_with_model(model, validation_bundle)
                validation_target = pd.Series(
                    validation_bundle.targets.detach().cpu().numpy(),
                    index=validation_bundle.index,
                    dtype=float,
                )
                validation_series = pd.Series(
                    validation_predictions,
                    index=validation_bundle.index,
                    dtype=float,
                )
                validation_ic = information_coefficient(validation_target, validation_series)
                scheduler.step(validation_ic if pd.notna(validation_ic) else float("-inf"))

                improved = pd.notna(validation_ic) and (
                    validation_ic > best_validation_ic + 1e-6
                )
                if improved:
                    best_validation_ic = float(validation_ic)
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.early_stopping_patience:
                        break
            else:
                best_epoch = epoch
                best_validation_ic = float("nan")
                best_state = copy.deepcopy(model.state_dict())

        if validation_bundle is not None and best_epoch == 0:
            best_epoch = epochs_run
            best_validation_ic = float("nan")
            best_state = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_state)
        final_learning_rate = float(optimizer.param_groups[0]["lr"])
        training_info = LSTMTrainingInfo(
            epochs_run=epochs_run,
            best_epoch=best_epoch or epochs_run,
            best_validation_ic=best_validation_ic,
            early_stopped=epochs_run < self.max_epochs,
            final_learning_rate=final_learning_rate,
            train_samples=train_bundle.sample_count,
            validation_samples=validation_bundle.sample_count if validation_bundle is not None else 0,
        )
        return {
            "model": model,
            "training_info": training_info,
            "best_validation_ic": best_validation_ic,
        }

    def _predict_bundle(self, bundle: _SequenceBundle) -> np.ndarray:
        if self.estimator_ is None:
            raise RuntimeError("LSTMModel must be trained before prediction.")
        return self._predict_bundle_with_model(self.estimator_, bundle)

    def _predict_bundle_with_model(
        self,
        model: _LSTMRegressor,
        bundle: _SequenceBundle,
    ) -> np.ndarray:
        loader = self._build_loader(bundle, batch_size=self.batch_size, shuffle=False)
        outputs: list[np.ndarray] = []
        model.eval()
        with torch.inference_mode():
            for batch in loader:
                if isinstance(batch, (tuple, list)):
                    batch_sequences = batch[0]
                else:
                    batch_sequences = batch
                batch_sequences = batch_sequences.to(self._torch_device())
                predictions = model(batch_sequences).detach().cpu().numpy().astype(np.float32, copy=False)
                outputs.append(predictions)
        if not outputs:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(outputs)

    def _build_loader(
        self,
        bundle: _SequenceBundle,
        *,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        if bundle.targets is None:
            dataset = TensorDataset(bundle.sequences)
        else:
            dataset = TensorDataset(bundle.sequences, bundle.targets)
        return DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=shuffle,
            drop_last=False,
        )

    def _build_sequence_bundle(
        self,
        features: pd.DataFrame,
        anchor_index: pd.Index,
        targets: pd.Series | None,
    ) -> _SequenceBundle:
        anchors = self._prepare_anchor_index(anchor_index)
        if targets is not None:
            target_series = self._prepare_target(targets).reindex(anchors)
            valid_target_mask = target_series.notna()
            anchors = anchors[valid_target_mask.to_numpy()]
            target_values = target_series.loc[anchors].to_numpy(dtype=np.float32, copy=True)
        else:
            target_values = None

        feature_frame = self._prepare_feature_frame(features, expected_columns=self.feature_names_ or None)
        self.feature_names_ = list(feature_frame.columns)
        feature_lookup = _build_feature_lookup(feature_frame)

        sample_count = len(anchors)
        if sample_count == 0:
            empty_index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=["trade_date", "ticker"])
            empty_sequences = torch.empty((0, self.sequence_length, len(self.feature_names_)), dtype=torch.float32)
            empty_targets = torch.empty((0,), dtype=torch.float32) if targets is not None else None
            return _SequenceBundle(sequences=empty_sequences, targets=empty_targets, index=empty_index)

        anchor_dates = pd.to_datetime(anchors.get_level_values(_date_level_name(anchors)))
        anchor_tickers = (
            anchors.get_level_values(_ticker_level_name(anchors)).astype(str).str.upper().tolist()
        )
        sequences = np.zeros(
            (sample_count, self.sequence_length, len(self.feature_names_)),
            dtype=np.float32,
        )
        valid_mask = np.zeros(sample_count, dtype=bool)

        for position, (trade_date, ticker) in enumerate(zip(anchor_dates, anchor_tickers, strict=False)):
            lookup = feature_lookup.get(str(ticker).upper())
            if lookup is None:
                continue
            anchor_pos = lookup["positions"].get(pd.Timestamp(trade_date))
            if anchor_pos is None:
                continue

            start_pos = max(0, anchor_pos - self.sequence_length + 1)
            history = lookup["values"][start_pos : anchor_pos + 1]
            sequences[position, -len(history) :, :] = history
            valid_mask[position] = True

        valid_positions = np.flatnonzero(valid_mask)
        if len(valid_positions) == 0:
            empty_index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=anchors.names)
            empty_sequences = torch.empty((0, self.sequence_length, len(self.feature_names_)), dtype=torch.float32)
            empty_targets = torch.empty((0,), dtype=torch.float32) if targets is not None else None
            return _SequenceBundle(sequences=empty_sequences, targets=empty_targets, index=empty_index)

        filtered_index = anchors[valid_positions]
        filtered_sequences = torch.from_numpy(sequences[valid_positions])
        filtered_targets = (
            torch.from_numpy(target_values[valid_positions])
            if target_values is not None
            else None
        )
        return _SequenceBundle(
            sequences=filtered_sequences,
            targets=filtered_targets,
            index=filtered_index,
        )

    def _temporal_holdout_split(
        self,
        target: pd.Series,
    ) -> tuple[pd.Series, pd.Series | None]:
        unique_dates = pd.Index(pd.to_datetime(target.index.get_level_values(_date_level_name(target.index))).unique()).sort_values()
        if len(unique_dates) < 10:
            return target.sort_index(), None

        validation_dates = max(1, int(math.ceil(len(unique_dates) * self.validation_fraction)))
        validation_dates = min(validation_dates, len(unique_dates) - 1)
        holdout_set = set(unique_dates[-validation_dates:])
        anchor_dates = pd.to_datetime(target.index.get_level_values(_date_level_name(target.index)))
        validation_mask = anchor_dates.isin(holdout_set)
        train_target = target.loc[~validation_mask].sort_index()
        validation_target = target.loc[validation_mask].sort_index()
        if train_target.empty or validation_target.empty:
            return target.sort_index(), None
        return train_target, validation_target

    def _prepare_feature_frame(
        self,
        X: pd.DataFrame,
        expected_columns: Sequence[str] | None,
    ) -> pd.DataFrame:
        features = self.validate_features(X)
        if not isinstance(features.index, pd.MultiIndex):
            raise ValueError("LSTMModel requires a MultiIndex(trade_date, ticker) feature frame.")

        cleaned = features.copy()
        cleaned.columns = [str(column) for column in cleaned.columns]
        cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
        cleaned = cleaned.fillna(0.0)
        cleaned.sort_index(inplace=True)

        ticker_level = _ticker_level_name(cleaned.index)
        cleaned.index = cleaned.index.set_levels(
            cleaned.index.levels[cleaned.index.names.index(ticker_level)].astype(str),
            level=ticker_level,
        ) if ticker_level in cleaned.index.names else cleaned.index

        if expected_columns is not None:
            missing_columns = [column for column in expected_columns if column not in cleaned.columns]
            if missing_columns:
                raise ValueError(f"LSTMModel feature frame is missing columns: {missing_columns}")
            cleaned = cleaned.loc[:, list(expected_columns)]
        return cleaned.astype(np.float32)

    def _prepare_target(self, y: pd.Series) -> pd.Series:
        target = self.validate_target(y).copy()
        if not isinstance(target.index, pd.MultiIndex):
            raise ValueError("LSTMModel requires a MultiIndex(trade_date, ticker) target series.")
        target = pd.to_numeric(target, errors="coerce")
        target = target.dropna().sort_index()
        if target.empty:
            raise ValueError("LSTMModel target series contains no non-null observations.")
        return target.astype(np.float32)

    def _prepare_anchor_index(self, anchor_index: pd.Index) -> pd.MultiIndex:
        if not isinstance(anchor_index, pd.MultiIndex):
            raise ValueError("LSTMModel anchor index must be a MultiIndex(trade_date, ticker).")
        return anchor_index.sort_values()

    def _configure_torch_runtime(self) -> None:
        torch.set_num_threads(max(1, self.n_jobs))

    def _torch_device(self) -> torch.device:
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _set_random_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _apply_hyperparameters(self, params: Mapping[str, Any]) -> None:
        for key, value in params.items():
            setattr(self, key, _normalize_scalar(value))

    def _current_hyperparameters(self) -> dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
        }


def _sample_lstm_candidates(
    *,
    search_space: Mapping[str, Sequence[Any]],
    n_iter: int,
    random_state: int,
) -> list[dict[str, Any]]:
    keys = list(search_space)
    values = [tuple(search_space[key]) for key in keys]
    grid = [dict(zip(keys, combination, strict=False)) for combination in product(*values)]
    if not grid:
        return [{}]
    if n_iter >= len(grid):
        return grid
    rng = np.random.default_rng(random_state)
    selected = rng.choice(len(grid), size=n_iter, replace=False)
    return [grid[int(index)] for index in selected]


def _build_feature_lookup(features: pd.DataFrame) -> dict[str, dict[str, Any]]:
    ticker_level = _ticker_level_name(features.index)
    lookup: dict[str, dict[str, Any]] = {}
    for ticker, group in features.groupby(level=ticker_level, sort=False):
        frame = group.droplevel(ticker_level)
        date_level = _date_level_name(frame.index) if isinstance(frame.index, pd.MultiIndex) else 0
        if isinstance(frame.index, pd.MultiIndex):
            dates = pd.to_datetime(frame.index.get_level_values(date_level))
        else:
            dates = pd.to_datetime(frame.index)
        date_index = pd.Index(dates, name="trade_date")
        lookup[str(ticker).upper()] = {
            "values": frame.to_numpy(dtype=np.float32, copy=True),
            "positions": {timestamp: position for position, timestamp in enumerate(date_index)},
        }
    return lookup


def _ticker_level_name(index: pd.MultiIndex) -> str | int:
    if "ticker" in index.names:
        return "ticker"
    if len(index.names) < 2:
        raise ValueError("LSTMModel requires a MultiIndex with a ticker level.")
    return 1


def _date_level_name(index: pd.MultiIndex) -> str | int:
    for candidate in ("date", "trade_date", "signal_date", "calc_date"):
        if candidate in index.names:
            return candidate
    return 0


def _normalize_scalar(value: Any) -> Any:
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            return value
    return value


def format_metric(value: float) -> str:
    if pd.isna(value):
        return "nan"
    return f"{value:.6f}"
