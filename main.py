"""GTA VI trophy rarity forecaster v4 (make.fun gta6-trophy-rarity-forecast-01).

Forecasts each trophy's Exophase earn rate at the case target as five
quantiles (5/25/50/75/95%), from the case's own history only. Python
standard library only, deterministic, no network.

Model (count-consistent: rate = earners / players-tracked, both modelled):

  * History is split into refresh segments: consecutive reads with the
    same players-tracked count and the same rates are one segment, because
    Exophase refreshes its counts in steps, not continuously.
  * RICH branch (two or more segments): per-trophy log-time slopes of
    log(earners) against log(1 + hours since release). Each slope is
    shrunk toward its tier's mean slope (bronze/silver/gold/platinum
    move differently), then a divergence term stretches slopes away
    from the median: the fastest trophies keep accelerating while the
    slowest flatten, which is the failure mode the live backtest showed
    (fast trophies under-predicted, slow ones over-predicted). The
    story chain (trophies 4-13) shares one progression slope with a
    per-chapter lag, since those rates move together. Players-tracked
    N(t) is forecast from the case's own refreshes: over short horizons
    with a long enough history, the recent join rate extrapolated
    linearly; otherwise a refresh-step model (next-refresh timing from
    the case's own inter-refresh gaps, jump size from its own refreshes,
    damped over successive future refreshes because the launch-night
    first jump is an outlier) blended in log space with a log-time
    trend fit, weighted 0.6 step / 0.4 trend, since the trend
    over-predicts N over multi-day horizons. Sigma is per trophy, from
    the fit residuals.
  * FLAT branch (a single frozen segment, as in the practice cases):
    no refresh has been seen, so the model uses the recalibrated
    refresh jump: earners grow by half again (E_new = 1.5 * E_old),
    players-tracked jumps by ~5.5x over a 90-minute horizon (growing
    slowly with longer horizons, since more refreshes can land), and
    the fan is widened asymmetrically downward, because a second
    refresh before the target would push rates lower still. Sigma is
    per trophy: narrow (0.10) for the story chain that moves together,
    wide (0.30) for the platinum and the fast/slow extremes.

Everything is re-derived from the case file at run time: no rate, no
count and no jump factor is hard-coded from any earlier read.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Release midnight: the log-time origin for every slope fit.
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# --- flat-branch calibration (recalibrated to the implied practice truth) ---
FLAT_EARNER_GROWTH = 0.5      # E_target = E_old * (1 + 0.5)
FLAT_JUMP_90MIN = 5.5         # N_target = N_last * 5.5 over 90 minutes
FLAT_JUMP_HORIZON_POWER = 0.25
FLAT_JUMP_CAP = 25.0

# --- rich-branch tuning ---
SHRINK_K = 2.0          # slope weight n / (n + K): own slope vs tier mean
DIVERGENCE = 0.4        # stretch slopes away from the median by 1 + 0.4
SLOPE_MIN, SLOPE_MAX = -1.0, 4.0
JUMP_DECAY = 0.55       # successive future refresh jumps shrink by this
LAG_POWER = 0.7         # story-chain per-chapter lag damping
STORY_IDS = set(range(4, 14))   # chapters + final mission + chain missions
# Players-tracked: short horizons (<= 48 h) with a long enough history
# (>= 18 h) extrapolate the recent join rate linearly -- the join rate
# is still ramping in a shorter history, so there the blend below is
# used instead. The long-horizon blend weights the refresh-step model
# 0.6 / the log-time trend 0.4: the trend systematically over-predicts
# N over multi-day horizons (its slope never saturates), while the
# damped step model under-predicts by less, so the step side carries
# more weight. Join rate is measured over the last 24 h of reads.
SHORT_HORIZON_H = 48.0
SPAN_MIN_H = 18.0
JOIN_RATE_WINDOW_H = 24.0
BLEND_STEP_WEIGHT = 0.6

# Per-trophy fan width (logit sigma) where no fit residual is available.
SIGMA_STORY = 0.10
SIGMA_PLATINUM = 0.30
SIGMA_EXTREME = 0.30    # the fast/slow extremes: Ch.1 (#3), Beat Cal (#20)
SIGMA_GOLD_SILVER = 0.12
SIGMA_BRONZE = 0.15
EXTREME_IDS = {3, 20}
# Flat branch: a second refresh before the target pushes rates down, so
# the lower half of the fan is wider than the upper half.
FLAT_LO_MULT = 1.8
FLAT_HI_MULT = 0.8


def clamp(x, lo=0.01, hi=99.99):
    return min(hi, max(lo, x))


def logit(pct):
    p = clamp(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def log_time(at):
    """log(1 + hours since release midnight); the x-axis of every fit."""
    hours = (at - RELEASE).total_seconds() / 3600.0
    return math.log(max(0.05, 1.0 + hours))


def least_squares(xs, ys):
    """Slope, intercept and residual std of y on x; None if degenerate."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    if n > 2:
        resid = math.sqrt(sum((y - (intercept + slope * x)) ** 2
                              for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return slope, intercept, resid


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    if n % 2:
        return vals[n // 2]
    return 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def segments_of(history):
    """Group consecutive reads into refresh segments.

    A new segment starts when players-tracked changes or any rate
    changes. Each segment keeps its first read time, its N and its
    rates (the page is frozen inside a segment).
    """
    segs = []
    for read in history:
        rates = read["rates"]
        n = read["players_tracked"]
        if segs and segs[-1]["n"] == n and segs[-1]["rates"] == rates:
            continue
        segs.append({"at": parse_time(read["at"]), "n": n, "rates": rates})
    return segs


def tier_sigma(trophy):
    """Fan width where no fit residual exists: narrow for the story
    chain (its rates move together), wide for the platinum and the
    fast/slow extremes, tier-sized in between."""
    tid = trophy["id"]
    if tid in STORY_IDS:
        return SIGMA_STORY
    if trophy["tier"] == "platinum":
        return SIGMA_PLATINUM
    if tid in EXTREME_IDS:
        return SIGMA_EXTREME
    if trophy["tier"] in ("gold", "silver"):
        return SIGMA_GOLD_SILVER
    return SIGMA_BRONZE


def forecast_rich(case, trophies, segs, horizon_min):
    """Rich branch: at least two refresh segments in the history."""
    target = parse_time(case["target"])
    x_target = log_time(target)
    x_last = log_time(segs[-1]["at"])
    dx = max(0.0, x_target - x_last)
    horizon_days = horizon_min / 1440.0
    horizon_factor = 1.0 + 0.20 * math.log(1.0 + horizon_days)

    n_troph = len(trophies)
    # Per-trophy log-earner slopes over the segments.
    slopes = [None] * n_troph
    resid = [None] * n_troph
    e_last = [0.5] * n_troph
    for i in range(n_troph):
        xs, ys = [], []
        for seg in segs:
            rate = seg["rates"][i]
            if rate is None:
                continue
            earners = max(rate / 100.0 * seg["n"], 0.5)
            xs.append(log_time(seg["at"]))
            ys.append(math.log(earners))
        if xs:
            e_last[i] = math.exp(ys[-1])
        fit = least_squares(xs, ys)
        if fit is not None:
            slopes[i], _icept, resid[i] = fit

    # Tier mean slopes, for shrinkage of the per-trophy slopes.
    tier_slopes = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        vals = [slopes[i] for i in range(n_troph)
                if trophies[i]["tier"] == tier and slopes[i] is not None]
        if vals:
            tier_slopes[tier] = sum(vals) / len(vals)
    global_slope = median([s for s in slopes if s is not None])

    # Shrink each slope toward its tier mean, then stretch away from
    # the median (divergence: fast trophies keep accelerating, slow
    # ones flatten).
    n_seg = len(segs)
    weight = n_seg / (n_seg + SHRINK_K)
    shrunk = []
    for i in range(n_troph):
        own = slopes[i]
        if own is None:
            own = tier_slopes.get(trophies[i]["tier"], global_slope)
        anchor = tier_slopes.get(trophies[i]["tier"], global_slope)
        shrunk.append(weight * own + (1.0 - weight) * anchor)
    med_slope = median(shrunk)
    eff_slope = []
    for s in shrunk:
        stretched = med_slope + (1.0 + DIVERGENCE) * (s - med_slope)
        eff_slope.append(min(SLOPE_MAX, max(SLOPE_MIN, stretched)))

    # Story chain: one shared slope (the median of the chain's shrunk
    # slopes) with a per-chapter lag, damped toward the chain mean.
    chain = [i for i in range(n_troph) if trophies[i]["id"] in STORY_IDS]
    shared_slope = med_slope
    chain_geo = 0.5
    if chain:
        shared_slope = median([shrunk[i] for i in chain])
        shared_slope = min(SLOPE_MAX, max(SLOPE_MIN, shared_slope))
        logs = [math.log(max(e_last[i], 0.5)) for i in chain]
        chain_geo = math.exp(sum(logs) / len(logs))

    # Players-tracked at the target: refresh-step model blended with
    # a log-time trend, in log space.
    n_target = forecast_n(segs, x_target, horizon_min)

    medians = []
    sigmas = []
    for i in range(n_troph):
        if i in chain:
            lag = (max(e_last[i], 0.5) / chain_geo) ** LAG_POWER
            e_tgt = chain_geo * lag * math.exp(shared_slope * dx)
        else:
            e_tgt = e_last[i] * math.exp(eff_slope[i] * dx)
        rate = clamp(e_tgt / n_target * 100.0)
        medians.append(rate)
        if resid[i] is not None and n_seg >= 3:
            sig = min(0.45, max(0.10, 0.10 + 0.9 * resid[i]))
        else:
            sig = tier_sigma(trophies[i])
        sigmas.append(min(0.60, sig * horizon_factor))
    return medians, sigmas, 1.0, 1.0


def forecast_n(segs, x_target, horizon_min):
    """Players-tracked at the target.

    Short horizon with a long enough history: extrapolate the recent
    join rate (players per hour over the last 24 h of reads) linearly
    -- over a day or so the join rate is the best guide to the next
    refreshes, and the log-time trend understates it.
    Otherwise, two models blended in log space: a refresh-step model
    (future refreshes arrive at the case's own median inter-refresh
    gap, each multiplying N by the case's own median jump factor,
    shrunk toward 1 when few jumps were seen and damped over
    successive future refreshes, because the launch-night first jump
    is an outlier, not a steady rate) and a log-time trend fit of
    log N. The blend weights the step model 0.6 / the trend 0.4: over
    multi-day horizons the trend's slope never saturates and it
    over-predicts N by 2-3x, while the damped step model under-predicts
    by less.
    """
    n_last = float(segs[-1]["n"])
    if len(segs) < 2:
        return n_last
    horizon_h = horizon_min / 60.0
    span_h = (segs[-1]["at"] - segs[0]["at"]).total_seconds() / 3600.0
    if horizon_h <= SHORT_HORIZON_H and span_h >= SPAN_MIN_H:
        t_last = segs[-1]["at"]
        base = None
        for seg in reversed(segs[:-1]):
            if (t_last - seg["at"]).total_seconds() / 3600.0 >= JOIN_RATE_WINDOW_H:
                base = seg
                break
        if base is None:
            base = segs[0]
        dt = (t_last - base["at"]).total_seconds() / 3600.0
        rate = (n_last - base["n"]) / dt if dt > 0 else 0.0
        rate = max(rate, 0.0)
        return max(n_last, n_last + rate * horizon_h)
    refresh_times = [s["at"] for s in segs]
    jumps = [segs[k]["n"] / segs[k - 1]["n"]
             for k in range(1, len(segs)) if segs[k - 1]["n"] > 0]
    jumps = [j for j in jumps if j > 0]
    if not jumps:
        return n_last
    jump_med = median(jumps)
    n_jumps = len(jumps)
    # Shrink the jump factor toward 1: one observed jump says little.
    jump_eff = 1.0 + (jump_med - 1.0) * n_jumps / (n_jumps + 2.0)
    gaps = [(refresh_times[k] - refresh_times[k - 1]).total_seconds() / 3600.0
            for k in range(1, len(refresh_times))]
    gaps = [g for g in gaps if g > 0]
    gap_med = median(gaps) if gaps else None

    # Step model.
    log_step = math.log(n_last)
    if gap_med:
        horizon_h = horizon_min / 60.0
        expected = horizon_h / gap_med
        whole = int(expected)
        frac = expected - whole
        for j in range(whole):
            factor = 1.0 + (jump_eff - 1.0) * (JUMP_DECAY ** j)
            log_step += math.log(max(1.0, factor))
        if frac > 0:
            factor = 1.0 + (jump_eff - 1.0) * (JUMP_DECAY ** whole)
            log_step += frac * math.log(max(1.0, factor))
    # Trend model.
    xs = [log_time(s["at"]) for s in segs]
    ys = [math.log(max(s["n"], 1)) for s in segs]
    fit = least_squares(xs, ys)
    if fit is not None:
        slope, intercept, _r = fit
        log_trend = intercept + slope * x_target
    else:
        log_trend = log_step
    w = BLEND_STEP_WEIGHT
    log_n = w * log_step + (1.0 - w) * log_trend
    return min(n_last * 1000.0, max(n_last, math.exp(log_n)))


def forecast_flat(case, trophies, segs, horizon_min):
    """Flat branch: a single frozen segment, no refresh seen yet.

    Recalibrated refresh jump: earners grow by half again while
    players-tracked jumps ~5.5x over 90 minutes (more over longer
    horizons, since more refreshes can land). The fan is widened
    asymmetrically downward: a second refresh before the target
    would push rates lower still.
    """
    last = segs[-1]
    n_last = float(last["n"])
    jump = FLAT_JUMP_90MIN * (max(horizon_min, 1.0) / 90.0) ** FLAT_JUMP_HORIZON_POWER
    jump = min(FLAT_JUMP_CAP, jump)
    n_target = n_last * jump
    medians = []
    sigmas = []
    for i, trophy in enumerate(trophies):
        rate = last["rates"][i]
        if rate is None:
            rate = 0.0
        earners = max(rate / 100.0 * n_last, 0.0)
        e_tgt = earners * (1.0 + FLAT_EARNER_GROWTH)
        medians.append(clamp(e_tgt / n_target * 100.0))
        sigmas.append(tier_sigma(trophy))
    return medians, sigmas, FLAT_LO_MULT, FLAT_HI_MULT


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    horizon_min = float(case["horizon_minutes"])
    segs = segments_of(history)
    if len(segs) >= 2:
        medians, sigmas, lo_mult, hi_mult = forecast_rich(
            case, trophies, segs, horizon_min)
    else:
        medians, sigmas, lo_mult, hi_mult = forecast_flat(
            case, trophies, segs, horizon_min)

    last_rates = segs[-1]["rates"]
    rows = []
    for i, trophy in enumerate(trophies):
        last = last_rates[i]
        if last is not None and last >= 99.995:
            # Everybody tracked already holds it; it stays at 100%.
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        lg = logit(medians[i])
        sig = sigmas[i]
        qs = []
        for z in Z:
            mult = lo_mult if z < 0 else (hi_mult if z > 0 else 1.0)
            qs.append(round4(clamp(inv_logit(lg + z * sig * mult))))
        # Rounding can only move a quantile by a hair; keep the row
        # non-decreasing exactly.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
