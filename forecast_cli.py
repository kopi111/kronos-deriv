"""Forecast Deriv candles with Kronos and report the resulting bias."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from deriv_data import fetch_candles, future_timestamps, granularity_label  # noqa: E402
from kronos_forecast import Forecast, forecast_candles, load_predictor  # noqa: E402

DEFAULT_SYMBOLS = ["R_10", "R_25", "R_50", "R_75", "R_100", "1HZ75V", "1HZ100V", "stpRNG"]
CHART_HISTORY_CANDLES = 96
OUTPUT_DIR = Path(__file__).resolve().parent / "out"


def main() -> None:
    arguments = parse_arguments()
    predictor = load_predictor(device=arguments.device)
    OUTPUT_DIR.mkdir(exist_ok=True)

    for symbol in arguments.symbols:
        candles = fetch_candles(symbol, arguments.granularity, arguments.lookback)
        forecast = build_forecast(predictor, candles, symbol, arguments)
        print_report(forecast)
        if arguments.chart:
            chart_path = save_chart(forecast, candles.tail(CHART_HISTORY_CANDLES))
            print(f"  chart          {chart_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kronos forecast for Deriv candles")
    parser.add_argument("--symbols", nargs="+", default=["R_75"], help=f"e.g. {' '.join(DEFAULT_SYMBOLS)}")
    parser.add_argument("--granularity", type=int, default=900, help="candle size in seconds")
    parser.add_argument("--lookback", type=int, default=512, help="candles of context (max 512 used)")
    parser.add_argument("--horizon", type=int, default=8, help="candles to forecast")
    parser.add_argument("--paths", type=int, default=20, help="independent sampled paths")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chart", action="store_true", help="write a PNG per symbol")
    return parser.parse_args()


def build_forecast(predictor, candles, symbol: str, arguments: argparse.Namespace) -> Forecast:
    stamps = future_timestamps(candles["timestamps"].iloc[-1], arguments.granularity, arguments.horizon)
    return forecast_candles(
        predictor=predictor,
        candles=candles,
        future_stamps=stamps,
        symbol=symbol,
        granularity=arguments.granularity,
        path_count=arguments.paths,
        temperature=arguments.temperature,
        top_p=arguments.top_p,
    )


def print_report(forecast: Forecast) -> None:
    timeframe = granularity_label(forecast.granularity)
    span = f"{forecast.horizon} x {timeframe}"
    print(f"\n{forecast.symbol}  ({timeframe} candles, {len(forecast.paths)} sampled paths)")
    print(f"  last close     {forecast.last_close:.5f}   @ {forecast.last_timestamp} UTC")
    print(f"  bias           {forecast.direction}  ({forecast.confidence * 100:.0f}% of paths) over {span}")
    print(f"  expected move  {forecast.expected_move_pct:+.3f}%")
    print(
        f"  close range    p10 {forecast.close_percentile(10):.5f}"
        f"   p50 {forecast.close_percentile(50):.5f}"
        f"   p90 {forecast.close_percentile(90):.5f}"
    )
    print(f"  projected high {forecast.projected_high:.5f}   projected low {forecast.projected_low:.5f}")


def save_chart(forecast: Forecast, history) -> Path:
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(history["timestamps"], history["close"], color="#1f77b4", linewidth=1.2, label="actual close")

    stamps = forecast.median_path.index
    closes = np.stack([path["close"].values for path in forecast.paths])
    axis.fill_between(
        stamps,
        np.percentile(closes, 10, axis=0),
        np.percentile(closes, 90, axis=0),
        color="#ff7f0e",
        alpha=0.2,
        label="p10-p90 band",
    )
    for path in forecast.paths:
        axis.plot(stamps, path["close"].values, color="#ff7f0e", alpha=0.15, linewidth=0.8)
    axis.plot(stamps, forecast.median_path["close"], color="#d62728", linewidth=1.8, label="median forecast")

    axis.axvline(forecast.last_timestamp, color="#888888", linestyle="--", linewidth=1)
    axis.set_title(
        f"{forecast.symbol} — Kronos {forecast.horizon}-candle forecast "
        f"({granularity_label(forecast.granularity)}), bias {forecast.direction} "
        f"{forecast.confidence * 100:.0f}%"
    )
    axis.legend(loc="upper left")
    axis.grid(alpha=0.2)
    figure.autofmt_xdate()

    chart_path = OUTPUT_DIR / f"{forecast.symbol}_{granularity_label(forecast.granularity)}.png"
    figure.savefig(chart_path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    return chart_path


if __name__ == "__main__":
    main()
