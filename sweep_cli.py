"""Run a walk-forward Kronos sweep and persist the sampled paths for scoring."""

import argparse

from kronos_forecast import load_predictor
from sweep import OUTPUT_DIR, run_sweep


def main() -> None:
    arguments = parse_arguments()
    OUTPUT_DIR.mkdir(exist_ok=True)
    predictor = load_predictor(device=arguments.device)

    for symbol in arguments.symbols:
        sweep = run_sweep(
            predictor=predictor,
            symbol=symbol,
            granularity=arguments.granularity,
            horizon=arguments.horizon,
            points=arguments.points,
            path_count=arguments.paths,
            stride=arguments.stride,
            batch_size=arguments.batch_size,
        )
        print(f"saved: {sweep.save()}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample Kronos forecasts walk-forward")
    parser.add_argument("--symbols", nargs="+", default=["frxXAUUSD"])
    parser.add_argument("--granularity", type=int, default=900)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--points", type=int, default=100)
    parser.add_argument("--paths", type=int, default=16)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    main()
