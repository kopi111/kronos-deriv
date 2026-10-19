"""Score Kronos direction calls from a saved sweep against naive baselines."""

import argparse
import json
from math import comb

import numpy as np

from sweep import OUTPUT_DIR, Sweep


def main() -> None:
    arguments = parse_arguments()
    report = {}

    for symbol in arguments.symbols:
        sweep = Sweep.load(symbol, arguments.granularity, arguments.horizon)
        report[symbol] = score_sweep(sweep)
        print_report(sweep, report[symbol])

    result_path = OUTPUT_DIR / f"direction_scores_h{arguments.horizon}.json"
    result_path.write_text(json.dumps(report, indent=2, default=float))
    print(f"\nwritten: {result_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Kronos direction calls")
    parser.add_argument("--symbols", nargs="+", default=["frxXAUUSD"])
    parser.add_argument("--granularity", type=int, default=900)
    parser.add_argument("--horizon", type=int, default=4)
    return parser.parse_args()


def score_sweep(sweep: Sweep) -> dict:
    last_closes = sweep.closes_before_cutoff()
    probabilities = (sweep.predicted_closes() > last_closes[:, None]).mean(axis=1)
    actuals = sweep.actual_closes() > last_closes
    calls = probabilities >= 0.5
    momentum = _momentum_directions(sweep)

    return {
        "points": sweep.point_count,
        "horizon": sweep.horizon,
        "kronos_hit_rate": _hit_rate(calls, actuals),
        "momentum_hit_rate": _hit_rate(momentum, actuals),
        "always_up_hit_rate": _hit_rate(np.ones_like(actuals, dtype=bool), actuals),
        "kronos_up_call_share": float(calls.mean()),
        "actual_up_share": float(actuals.mean()),
        "confidence_buckets": _confidence_buckets(probabilities, actuals),
        "binomial_p_value": _binomial_p_value(_hit_rate(calls, actuals), len(actuals)),
    }


def _momentum_directions(sweep: Sweep) -> np.ndarray:
    closes = sweep.candles["close"].values
    return np.array(
        [closes[cutoff - 1] > closes[cutoff - 1 - sweep.horizon] for cutoff in sweep.cutoffs]
    )


def _hit_rate(calls: np.ndarray, actuals: np.ndarray) -> float:
    return float((calls == actuals).mean())


def _confidence_buckets(probabilities: np.ndarray, actuals: np.ndarray) -> dict:
    confidence = np.maximum(probabilities, 1 - probabilities)
    calls = probabilities >= 0.5
    buckets = {}
    for low, high in [(0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]:
        selected = (confidence >= low) & (confidence < high)
        label = f"{int(low * 100)}-{int(min(high, 1.0) * 100)}%"
        buckets[label] = {
            "count": int(selected.sum()),
            "hit_rate": _hit_rate(calls[selected], actuals[selected]) if selected.any() else None,
        }
    return buckets


def _binomial_p_value(hit_rate: float, trials: int) -> float:
    """Two-sided probability of a hit rate this far from 50% under a fair coin."""
    hits = round(hit_rate * trials)
    deviation = abs(hits - trials / 2)
    tail = sum(comb(trials, k) for k in range(trials + 1) if abs(k - trials / 2) >= deviation)
    return float(tail / 2**trials)


def print_report(sweep: Sweep, result: dict) -> None:
    print(f"\n{sweep.symbol} — direction ({result['points']} points, "
          f"{result['horizon']} candles ahead)")
    print(f"  kronos     {result['kronos_hit_rate'] * 100:5.1f}%   p={result['binomial_p_value']:.3f}")
    print(f"  momentum   {result['momentum_hit_rate'] * 100:5.1f}%")
    print(f"  always up  {result['always_up_hit_rate'] * 100:5.1f}%")
    print(f"  kronos called UP on {result['kronos_up_call_share'] * 100:.0f}% of points, "
          f"actual up {result['actual_up_share'] * 100:.0f}%")
    for label, bucket in result["confidence_buckets"].items():
        if bucket["count"]:
            print(f"  conf {label:>8}  n={bucket['count']:<3} hit={bucket['hit_rate'] * 100:.0f}%")


if __name__ == "__main__":
    main()
