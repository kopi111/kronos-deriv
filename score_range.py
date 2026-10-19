"""Score how well Kronos predicts how far price moves, rather than which way."""

import argparse
import json

import numpy as np
from scipy.stats import spearmanr

from sweep import OUTPUT_DIR, Sweep

VOLATILITY_LOOKBACK = 100
BASELINE_SEED = 20260726


def main() -> None:
    arguments = parse_arguments()
    report = {}

    for symbol in arguments.symbols:
        sweep = Sweep.load(symbol, arguments.granularity, arguments.horizon)
        report[symbol] = score_sweep(sweep)
        print_report(sweep, report[symbol])

    result_path = OUTPUT_DIR / f"range_scores_h{arguments.horizon}.json"
    result_path.write_text(json.dumps(report, indent=2, default=float))
    print(f"\nwritten: {result_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Kronos range/interval forecasts")
    parser.add_argument("--symbols", nargs="+", default=["frxXAUUSD"])
    parser.add_argument("--granularity", type=int, default=900)
    parser.add_argument("--horizon", type=int, default=4)
    return parser.parse_args()


def score_sweep(sweep: Sweep) -> dict:
    predicted = sweep.predicted_closes()
    actual = sweep.actual_closes()
    baseline = _random_walk_samples(sweep)

    return {
        "points": sweep.point_count,
        "paths": sweep.path_count,
        "horizon": sweep.horizon,
        "interval_coverage": {
            "kronos": _coverage_table(predicted, actual),
            "random_walk": _coverage_table(baseline, actual),
        },
        "crps": {
            "kronos": _mean_crps(predicted, actual),
            "random_walk": _mean_crps(baseline, actual),
        },
        "crps_pct_of_price": {
            "kronos": _mean_crps(predicted, actual) / actual.mean() * 100,
            "random_walk": _mean_crps(baseline, actual) / actual.mean() * 100,
        },
        "band_width": _band_width_comparison(predicted, baseline),
        "realized_range": _range_comparison(sweep),
    }


def _coverage_table(samples: np.ndarray, actual: np.ndarray) -> dict:
    return {
        "p10_p90_target_80": _coverage(samples, actual, 10, 90),
        "p25_p75_target_50": _coverage(samples, actual, 25, 75),
        "p05_p95_target_90": _coverage(samples, actual, 5, 95),
    }


def _coverage(samples: np.ndarray, actual: np.ndarray, low: float, high: float) -> float:
    lower = np.percentile(samples, low, axis=1)
    upper = np.percentile(samples, high, axis=1)
    return float(((actual >= lower) & (actual <= upper)).mean() * 100)


def _mean_crps(samples: np.ndarray, actual: np.ndarray) -> float:
    """Continuous ranked probability score, estimated from samples. Lower is better."""
    scores = []
    for point_samples, observation in zip(samples, actual):
        accuracy = np.abs(point_samples - observation).mean()
        spread = np.abs(point_samples[:, None] - point_samples[None, :]).mean()
        scores.append(accuracy - 0.5 * spread)
    return float(np.mean(scores))


def _band_width_comparison(predicted: np.ndarray, baseline: np.ndarray) -> dict:
    kronos_width = np.percentile(predicted, 90, axis=1) - np.percentile(predicted, 10, axis=1)
    baseline_width = np.percentile(baseline, 90, axis=1) - np.percentile(baseline, 10, axis=1)
    return {
        "kronos_mean_p10_p90": float(kronos_width.mean()),
        "random_walk_mean_p10_p90": float(baseline_width.mean()),
        "kronos_relative_width": float(kronos_width.mean() / baseline_width.mean()),
    }


def _range_comparison(sweep: Sweep) -> dict:
    predicted = np.median(sweep.predicted_ranges(), axis=1)
    actual = sweep.actual_ranges()
    baseline = _atr_baseline_range(sweep)

    return {
        "kronos_spearman": float(spearmanr(predicted, actual).statistic),
        "atr_spearman": float(spearmanr(baseline, actual).statistic),
        "kronos_mean_predicted": float(predicted.mean()),
        "atr_mean_predicted": float(baseline.mean()),
        "mean_actual": float(actual.mean()),
        "kronos_bias": float(predicted.mean() / actual.mean()),
        "atr_bias": float(baseline.mean() / actual.mean()),
        "kronos_scaled_mae": _scaled_mae(predicted, actual),
        "atr_scaled_mae": _scaled_mae(baseline, actual),
    }


def _scaled_mae(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Mean absolute error after removing each estimator's constant bias.

    Both estimators get the same favour, so this compares shape rather than calibration.
    """
    scaled = predicted * (actual.mean() / predicted.mean())
    return float(np.abs(scaled - actual).mean())


def _random_walk_samples(sweep: Sweep) -> np.ndarray:
    """Closes drawn from a random walk using volatility trailing each cutoff."""
    generator = np.random.default_rng(BASELINE_SEED)
    closes = sweep.candles["close"].values
    last_closes = sweep.closes_before_cutoff()

    samples = []
    for cutoff, last_close in zip(sweep.cutoffs, last_closes):
        window = closes[cutoff - VOLATILITY_LOOKBACK : cutoff]
        sigma = np.std(np.diff(np.log(window)))
        shocks = generator.normal(0.0, sigma * np.sqrt(sweep.horizon), sweep.path_count)
        samples.append(last_close * np.exp(shocks))
    return np.array(samples)


def _atr_baseline_range(sweep: Sweep) -> np.ndarray:
    """Trailing average true range scaled to the forecast horizon."""
    frame = sweep.candles
    high, low, close = frame["high"].values, frame["low"].values, frame["close"].values
    true_range = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
    )
    return np.array(
        [true_range[cutoff - 15 : cutoff - 1].mean() * np.sqrt(sweep.horizon) for cutoff in sweep.cutoffs]
    )


def print_report(sweep: Sweep, result: dict) -> None:
    print(f"\n{sweep.symbol} — range accuracy ({result['points']} points, "
          f"{result['paths']} paths, {result['horizon']} candles ahead)")

    print("\n  interval coverage (want actual close inside the band at the target rate)")
    for label, target in [("p10_p90_target_80", 80), ("p25_p75_target_50", 50), ("p05_p95_target_90", 90)]:
        kronos = result["interval_coverage"]["kronos"][label]
        walk = result["interval_coverage"]["random_walk"][label]
        print(f"    target {target:>2}%   kronos {kronos:5.1f}%   random walk {walk:5.1f}%")

    print("\n  CRPS, lower is better (whole predicted distribution vs what happened)")
    print(f"    kronos       {result['crps']['kronos']:.4f}"
          f"   ({result['crps_pct_of_price']['kronos']:.4f}% of price)")
    print(f"    random walk  {result['crps']['random_walk']:.4f}"
          f"   ({result['crps_pct_of_price']['random_walk']:.4f}% of price)")
    edge = (1 - result["crps"]["kronos"] / result["crps"]["random_walk"]) * 100
    print(f"    kronos is {edge:+.1f}% vs the random-walk baseline")

    width = result["band_width"]
    print(f"\n  mean p10-p90 width   kronos {width['kronos_mean_p10_p90']:.3f}"
          f"   random walk {width['random_walk_mean_p10_p90']:.3f}"
          f"   (ratio {width['kronos_relative_width']:.2f})")

    realized = result["realized_range"]
    print("\n  realized high-low range over the horizon")
    print(f"    spearman vs actual   kronos {realized['kronos_spearman']:+.3f}"
          f"   ATR {realized['atr_spearman']:+.3f}")
    print(f"    mean predicted       kronos {realized['kronos_mean_predicted']:.3f}"
          f"   ATR {realized['atr_mean_predicted']:.3f}"
          f"   actual {realized['mean_actual']:.3f}")
    print(f"    bias                 kronos {realized['kronos_bias']:.2f}x"
          f"   ATR {realized['atr_bias']:.2f}x")
    print(f"    scaled MAE           kronos {realized['kronos_scaled_mae']:.3f}"
          f"   ATR {realized['atr_scaled_mae']:.3f}")


if __name__ == "__main__":
    main()
