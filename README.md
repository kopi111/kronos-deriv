# Kronos × Deriv

[Kronos](https://github.com/shiyu-coder/Kronos) — the open-source foundation model for candlestick
(K-line) data — wired to live Deriv WebSocket candles.

## Layout

| File | Purpose |
|---|---|
| `deriv_data.py` | Candle retrieval from the Deriv WebSocket API (no auth required) |
| `kronos_forecast.py` | Model loading, sampled-path forecasting, signal derivation |
| `forecast_cli.py` | Forecast one or more symbols, print the bias, optionally chart it |
| `validate.py` | Walk-forward test of whether the direction calls beat naive baselines |
| `Kronos/` | Upstream clone (MIT) |
| `out/` | Charts and validation JSON |

## Setup

Already done on this machine: `einops`, `huggingface_hub`, `safetensors` and `tqdm` are installed
into the user site-packages (`pip install --user --break-system-packages`), and
`Kronos-Tokenizer-base` + `Kronos-small` (24.7M params) are cached under `~/.cache/huggingface`.

Torch here is CPU-only, so `Kronos-small` is the practical ceiling. `Kronos-base` (102M) runs but is
roughly four times slower.

## Forecast

```bash
python3 forecast_cli.py --symbols R_75 --granularity 900 --horizon 8 --paths 20 --chart
```

```
R_75  (15m candles, 20 sampled paths)
  last close     46607.59580   @ 2026-07-26 20:45:00 UTC
  bias           UP  (90% of paths) over 8 x 15m
  expected move  +0.565%
  close range    p10 46612.83203   p50 46934.05078   p90 47155.69922
  projected high 47122.49609   projected low 46468.25781
```

Kronos is generative, not a classifier. Each sampled path is one plausible future; the bias is the
share of paths that close above the current price, and the p10–p90 spread is the model's own
uncertainty. `--paths 20` on CPU takes about a minute per symbol.

Deriv candles carry no volume, which Kronos accepts — it zero-fills the volume and amount columns.

## Validate before trusting it

```bash
python3 validate.py --symbols R_75 frxEURUSD --points 30 --paths 8 --horizon 4
```

Scores each direction call against what actually happened, alongside a momentum baseline and an
always-up baseline, with a two-sided binomial p-value for the hit rate. Results land in
`out/validation_<timeframe>.json`.

### Measured result — gold, 2026-07-26

60 non-overlapping points on `frxXAUUSD` 1h candles, 4 candles ahead, 8 sampled paths each:

| | hit rate |
|---|---|
| Kronos | 51.7% (p=0.897) |
| momentum | 45.0% |
| always up | 45.0% |

No edge. 51.7% over 60 points is what a coin flip looks like. Two further observations:

- **The model is systematically long.** It called UP on 70% of points while price actually rose on
  45% of them. Both live forecasts also came back "UP 90%".
- **Its confidence is not calibrated.** The 90–100% confidence bucket hit 40% (n=5), worse than the
  70–90% bucket at 64% (n=22). Higher confidence did not mean higher accuracy.

Sample sizes per bucket are small and this covers ~10 trading days of one regime, so this is a smell
test, not a verdict. But nothing here justifies trading the direction call.

## Known limits

- **Context is 512 candles.** On 15m bars that is ~5 days of history.
- **Deriv volatility indices are synthetic.** R_10/R_25/R_75/R_100 and the 1s variants are generated
  random walks with fixed volatility — no order flow, no participants, no regime. Kronos was
  pre-trained on real exchanges, so on synthetics it pattern-matches trend continuation that is not
  actually there. Treat a confident call on R_* as a prior about the model, not about the market.
- **`frx*` symbols carry real market data** and are the fairer test of whether the model adds
  anything.
