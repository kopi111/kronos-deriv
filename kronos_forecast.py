"""Probabilistic candle forecasting with the Kronos foundation model."""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

KRONOS_REPO = Path(__file__).resolve().parent / "Kronos"
sys.path.insert(0, str(KRONOS_REPO))

from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402

TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_NAME = "NeoQuasar/Kronos-small"
MAX_CONTEXT = 512


@dataclass(frozen=True)
class Forecast:
    """A sampled distribution of future candles and the signal derived from it."""

    symbol: str
    granularity: int
    horizon: int
    last_close: float
    last_timestamp: pd.Timestamp
    paths: list[pd.DataFrame]
    median_path: pd.DataFrame

    @property
    def final_closes(self) -> np.ndarray:
        return np.array([path["close"].iloc[-1] for path in self.paths])

    @property
    def probability_up(self) -> float:
        return float((self.final_closes > self.last_close).mean())

    @property
    def expected_move_pct(self) -> float:
        return float((self.final_closes.mean() / self.last_close - 1) * 100)

    @property
    def direction(self) -> str:
        return "UP" if self.probability_up >= 0.5 else "DOWN"

    @property
    def confidence(self) -> float:
        return max(self.probability_up, 1 - self.probability_up)

    def close_percentile(self, percentile: float) -> float:
        return float(np.percentile(self.final_closes, percentile))

    @property
    def projected_high(self) -> float:
        return float(np.median([path["high"].max() for path in self.paths]))

    @property
    def projected_low(self) -> float:
        return float(np.median([path["low"].min() for path in self.paths]))


def load_predictor(device: str = "cpu", model_name: str = MODEL_NAME) -> KronosPredictor:
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
    model = Kronos.from_pretrained(model_name)
    return KronosPredictor(model, tokenizer, device=device, max_context=MAX_CONTEXT)


def forecast_candles(
    predictor: KronosPredictor,
    candles: pd.DataFrame,
    future_stamps: pd.Series,
    symbol: str,
    granularity: int,
    path_count: int,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> Forecast:
    """Sample `path_count` independent future candle paths and summarise them."""
    context = candles.tail(MAX_CONTEXT).reset_index(drop=True)
    history = context[["open", "high", "low", "close"]]
    history_stamps = context["timestamps"]
    horizon = len(future_stamps)

    paths = predictor.predict_batch(
        df_list=[history.copy() for _ in range(path_count)],
        x_timestamp_list=[history_stamps.copy() for _ in range(path_count)],
        y_timestamp_list=[future_stamps.copy() for _ in range(path_count)],
        pred_len=horizon,
        T=temperature,
        top_p=top_p,
        sample_count=1,
        verbose=False,
    )

    return Forecast(
        symbol=symbol,
        granularity=granularity,
        horizon=horizon,
        last_close=float(context["close"].iloc[-1]),
        last_timestamp=context["timestamps"].iloc[-1],
        paths=paths,
        median_path=_median_path(paths, future_stamps),
    )


def _median_path(paths: list[pd.DataFrame], future_stamps: pd.Series) -> pd.DataFrame:
    stacked = np.stack([path[["open", "high", "low", "close"]].values for path in paths])
    median = np.median(stacked, axis=0)
    return pd.DataFrame(median, columns=["open", "high", "low", "close"], index=future_stamps)
