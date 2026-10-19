"""Walk-forward sampling of Kronos forecasts, persisted so scorers need no further inference."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from deriv_data import fetch_candles, future_timestamps, granularity_label
from kronos_forecast import MAX_CONTEXT

OUTPUT_DIR = Path(__file__).resolve().parent / "out"
PRICE_COLUMNS = ["open", "high", "low", "close"]


class InsufficientHistoryError(Exception):
    """Raised when the available history cannot support the requested evaluation points."""


@dataclass(frozen=True)
class Sweep:
    """Sampled forecast paths for a set of walk-forward cutoffs, plus the candles they came from."""

    symbol: str
    granularity: int
    horizon: int
    cutoffs: np.ndarray
    paths: np.ndarray  # (points, paths, horizon, 4) in open/high/low/close order
    candles: pd.DataFrame

    @property
    def point_count(self) -> int:
        return len(self.cutoffs)

    @property
    def path_count(self) -> int:
        return self.paths.shape[1]

    def closes_before_cutoff(self) -> np.ndarray:
        return self.candles["close"].values[self.cutoffs - 1]

    def predicted_closes(self) -> np.ndarray:
        """Final close of every sampled path, shaped (points, paths)."""
        return self.paths[:, :, -1, PRICE_COLUMNS.index("close")]

    def predicted_ranges(self) -> np.ndarray:
        """High-low span of every sampled path, shaped (points, paths)."""
        highs = self.paths[:, :, :, PRICE_COLUMNS.index("high")].max(axis=2)
        lows = self.paths[:, :, :, PRICE_COLUMNS.index("low")].min(axis=2)
        return highs - lows

    def actual_closes(self) -> np.ndarray:
        return self.candles["close"].values[self.cutoffs + self.horizon - 1]

    def actual_ranges(self) -> np.ndarray:
        highs = np.array(
            [self.candles["high"].values[c : c + self.horizon].max() for c in self.cutoffs]
        )
        lows = np.array(
            [self.candles["low"].values[c : c + self.horizon].min() for c in self.cutoffs]
        )
        return highs - lows

    def save(self) -> Path:
        path = self.file_path(self.symbol, self.granularity, self.horizon)
        np.savez_compressed(
            path,
            symbol=self.symbol,
            granularity=self.granularity,
            horizon=self.horizon,
            cutoffs=self.cutoffs,
            paths=self.paths,
            candles=self.candles.to_records(index=False),
        )
        return path

    @staticmethod
    def file_path(symbol: str, granularity: int, horizon: int) -> Path:
        return OUTPUT_DIR / f"sweep_{symbol}_{granularity_label(granularity)}_h{horizon}.npz"

    @classmethod
    def load(cls, symbol: str, granularity: int, horizon: int) -> "Sweep":
        path = cls.file_path(symbol, granularity, horizon)
        if not path.exists():
            raise FileNotFoundError(f"no sweep at {path} — run sweep.py first")
        stored = np.load(path, allow_pickle=True)
        return cls(
            symbol=str(stored["symbol"]),
            granularity=int(stored["granularity"]),
            horizon=int(stored["horizon"]),
            cutoffs=stored["cutoffs"],
            paths=stored["paths"],
            candles=pd.DataFrame.from_records(stored["candles"]),
        )


def run_sweep(
    predictor,
    symbol: str,
    granularity: int,
    horizon: int,
    points: int,
    path_count: int,
    stride: int,
    batch_size: int,
) -> Sweep:
    candles = fetch_candles(symbol, granularity, _required_candles(points, stride, horizon))
    cutoffs = _evaluation_cutoffs(candles, points, stride, horizon)
    print(f"{symbol}: sampling {points} points x {path_count} paths on {len(candles)} candles")

    items = [(cutoff, index) for cutoff in cutoffs for index in range(path_count)]
    sampled: dict[int, list[np.ndarray]] = {cutoff: [] for cutoff in cutoffs}

    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        for (cutoff, _), prediction in zip(
            chunk, _predict_chunk(predictor, candles, chunk, granularity, horizon)
        ):
            sampled[cutoff].append(prediction[PRICE_COLUMNS].values)
        print(f"  {min(start + len(chunk), len(items))}/{len(items)} paths", flush=True)

    paths = np.stack([np.stack(sampled[cutoff]) for cutoff in cutoffs])
    return Sweep(symbol, granularity, horizon, np.array(cutoffs), paths, candles)


def _required_candles(points: int, stride: int, horizon: int) -> int:
    return MAX_CONTEXT + stride * points + horizon + 1


def _evaluation_cutoffs(candles: pd.DataFrame, points: int, stride: int, horizon: int) -> list[int]:
    last_usable = len(candles) - horizon
    cutoffs = list(range(MAX_CONTEXT, last_usable, stride))[-points:]
    if len(cutoffs) < points:
        raise InsufficientHistoryError(
            f"asked for {points} evaluation points but only {len(cutoffs)} fit in "
            f"{len(candles)} candles (needs {_required_candles(points, stride, horizon)})"
        )
    return cutoffs


def _predict_chunk(predictor, candles, chunk, granularity: int, horizon: int) -> list[pd.DataFrame]:
    histories, history_stamps, future_stamps = [], [], []
    for cutoff, _ in chunk:
        context = candles.iloc[cutoff - MAX_CONTEXT : cutoff]
        histories.append(context[PRICE_COLUMNS].reset_index(drop=True))
        history_stamps.append(context["timestamps"].reset_index(drop=True))
        future_stamps.append(future_timestamps(context["timestamps"].iloc[-1], granularity, horizon))

    return predictor.predict_batch(
        df_list=histories,
        x_timestamp_list=history_stamps,
        y_timestamp_list=future_stamps,
        pred_len=horizon,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=False,
    )
