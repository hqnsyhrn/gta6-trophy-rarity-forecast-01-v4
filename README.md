[![Competition](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hqnsyhrn/gta6-trophy-rarity-forecast-01-v4/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hqnsyhrn/gta6-trophy-rarity-forecast-01-v4/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecaster v4

Forecasts, for every Grand Theft Auto VI trophy on PlayStation 5, its earn
rate on Exophase at the case target, as five quantiles (5/25/50/75/95%) per
trophy. An entry for the [make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
"Forecast how many GTA VI players earn each trophy in week one" ($250, top 20).

## Run

```sh
python3 main.py
```

The program reads every `<id>.in` case in `$MAKEFUN_INPUT_DIR` (default
`/input`) and writes `<id>.out` to `$MAKEFUN_OUTPUT_DIR` (default
`/output`). Python standard library only, deterministic, no network.

## Model

Count-consistent: each rate is earners divided by players tracked, and both
are modelled from the case file at run time — nothing is hard-coded from any
earlier read.

- The history is split into refresh segments, because Exophase refreshes its
  counts in steps, not continuously.
- Rich branch (two or more segments): per-trophy log-time slopes of
  log(earners), shrunk toward the trophy's tier mean, plus a divergence term
  that lets the fastest trophies keep accelerating while the slowest flatten.
  The story chain (trophies 4–13) shares one progression slope with a
  per-chapter lag. Players tracked is forecast from the case's own refreshes:
  the recent join rate over short horizons, otherwise a refresh-step model
  blended with a log-time trend (0.6 step / 0.4 trend). Sigma is per trophy,
  from the fit residuals.
- Flat branch (one frozen segment, as in the practice cases): the recalibrated
  refresh jump — earners grow by half again while players tracked jumps ~5.5x
  over 90 minutes — with the fan widened asymmetrically downward, since a
  second refresh before the target would push rates lower still. Sigma is per
  trophy: narrow for the story chain, wide for the platinum and the fast/slow
  extremes.

## Validation (local, before submit)

- Practice p00/p01 vs interpolated truth: loss 0.0667 / 0.0667, miss 0.027pp,
  coverage 0.95.
- Live backtest (history 04:30Z 19 Nov → live page 07:41Z 20 Nov): loss 1.4222
  vs persistence 2.2803. On the earlier 05:14Z target the same setup scores
  0.2931 vs v3's 0.3193.
- Synthetic worlds A/B (rollout + final): 0.1079 / 0.1229 / 0.2099 / 0.1213,
  beating persistence in all four (0.1148 / 0.2746 / 0.4507 / 0.1790).
- Deterministic: repeated runs on p00 are byte-identical; ~0.07s per case.

## License

MIT. See [LICENSE](LICENSE).
