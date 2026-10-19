"""Candle retrieval from the Deriv WebSocket API."""

import asyncio
import json
from datetime import datetime, timezone

import pandas as pd
import websockets

DERIV_APP_ID = 1089
DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"
MAX_CANDLES_PER_REQUEST = 5000

GRANULARITY_LABELS = {
    60: "1m",
    120: "2m",
    180: "3m",
    300: "5m",
    600: "10m",
    900: "15m",
    1800: "30m",
    3600: "1h",
    7200: "2h",
    14400: "4h",
    28800: "8h",
    86400: "1d",
}


class DerivDataError(Exception):
    """Raised when Deriv returns an error or an unusable response."""


def granularity_label(granularity_seconds: int) -> str:
    return GRANULARITY_LABELS.get(granularity_seconds, f"{granularity_seconds}s")


def fetch_candles(symbol: str, granularity: int, count: int) -> pd.DataFrame:
    """Return up to the most recent `count` candles for `symbol` as an OHLC DataFrame.

    Deriv treats the request `count` as a time window rather than a candle count, so a market
    that closes overnight and at weekends returns far fewer candles than asked for. This pages
    backwards until `count` candles are collected or the history runs out.
    """
    return asyncio.run(_request_candles(symbol, granularity, count))


async def _request_candles(symbol: str, granularity: int, count: int) -> pd.DataFrame:
    collected: list[dict] = []
    end = "latest"

    async with websockets.connect(DERIV_WS_URL) as socket:
        while len(collected) < count:
            page = await _request_page(socket, symbol, granularity, end)
            if not page:
                break
            collected = page + collected
            end = str(page[0]["epoch"] - 1)

    if not collected:
        raise DerivDataError(f"{symbol}: Deriv returned no candles")
    return _to_dataframe(collected).tail(count).reset_index(drop=True)


async def _request_page(socket, symbol: str, granularity: int, end: str) -> list[dict]:
    request = {
        "ticks_history": symbol,
        "adjust_start_time": 1,
        "count": MAX_CANDLES_PER_REQUEST,
        "end": end,
        "style": "candles",
        "granularity": granularity,
    }
    await socket.send(json.dumps(request))
    response = json.loads(await asyncio.wait_for(socket.recv(), timeout=30))

    if "error" in response:
        raise DerivDataError(f"{symbol}: {response['error'].get('message', response['error'])}")
    return response.get("candles", [])


def _to_dataframe(candles: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(candles).drop_duplicates(subset="epoch")
    frame["timestamps"] = pd.to_datetime(frame["epoch"], unit="s", utc=True).dt.tz_localize(None)
    frame = frame[["timestamps", "open", "high", "low", "close"]].astype(
        {"open": float, "high": float, "low": float, "close": float}
    )
    return frame.sort_values("timestamps").reset_index(drop=True)


def future_timestamps(last_timestamp: pd.Timestamp, granularity: int, horizon: int) -> pd.Series:
    """Return the `horizon` candle timestamps that follow `last_timestamp`."""
    step = pd.Timedelta(seconds=granularity)
    return pd.Series([last_timestamp + step * (i + 1) for i in range(horizon)])


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
