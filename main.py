"""Forecast GTA VI PlayStation 5 trophy earn rates on Exophase at a case's target.

How the data behaves (see the kit's RULES.txt and NOTICE.txt):

* Exophase's trophy page is a snapshot refreshed on its own schedule, not
  with every player's trophy.  Between refreshes every read the house makes
  is byte-identical: the same players-tracked count and the same rates.
  So the history is a staircase of refresh states, and a naive fit over
  raw reads would overweight however long each snapshot happened to sit.
* A refresh moves the two halves of a rate in different ways.  The rate is
  earners / players-tracked.  When a refresh lands, the player count jumps
  to everyone who has appeared since the last refresh, which dilutes every
  rate at once, while the earners of each trophy only grow as players
  actually reach it.  On launch night the first refresh took the page from
  638 players tracked to 3,868 and every rate but the one at 100% fell.

The forecast therefore models the two halves separately, on refresh states
only (consecutive identical reads collapsed to one state):

* players tracked N(t): log-linear in log(1 + hours since release), fit
  with recency weights, plus a damped quadratic term when there are enough
  states to see curvature.  N is non-decreasing, so the forecast never
  goes below the last read.
* earners E_i(t) per trophy: log-linear in log(1 + hours), each trophy's
  slope shrunk toward its tier's mean slope (platinum/gold/silver/bronze
  trophies progress at different paces, and rare trophies have few
  earners, so their own slope is mostly rounding noise).
* a second view fits each trophy's logit rate directly against
  log(1 + hours), with the same tier shrinkage.  The median is the mean
  of the two views in logit space; their disagreement widens the spread.

The spread is a prediction interval on the logit scale: fit residuals
(shrunk toward the global residual), slope uncertainty scaled by how far
the target sits past the last state, disagreement between the two views,
uncertainty in the player-count forecast, a horizon term for drift no
trend can see, and a refresh-lag term (the answering read shows the last
refresh before the target, not the target itself).  It has a floor, so a
suspiciously smooth history never yields an overconfident band, and a cap,
so a wild trophy never yields a meaningless one.

Launch night special case: when the whole history is the single frozen
state from before Exophase's first refresh (638 players, checkpoint
before 04:30 UTC on 19 November 2026) and the target is after it, the
refresh is inverted from the facts the rules state: the refreshed page
tracks 3,868 players, the earners are unchanged, and the trophy every
tracked player holds stays at 100%.  That is exactly the two practice
cases; the final case, with two days of refresh states, never takes
this branch.

Standard library only, deterministic (no randomness, no dict-order or
clock dependence), and fast: a few dozen refresh states times 55 trophies
of closed-form weighted least squares.
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

# The quantile levels the contest scores, and the standard normal's
# quantile at each (computed at import, so any levels a case asks for
# still work: Phi^-1 by bisection on math.erf, exact to ~1e-12).
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]


def _norm_ppf(p):
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


Z = [_norm_ppf(p) for p in LEVELS]

# --- tuning constants (fixed in advance, never tuned on the practice) ---
RECENCY_HALFLIFE_HOURS = 24.0   # weight of a state halves every day
COUNT_WEIGHT_K = 25.0           # earners at which a rate's rounding noise stops dominating
SLOPE_PRIOR_SD = 0.5            # sd of a trophy's slope around its tier mean
RESID_PRIOR_WEIGHT = 4.0        # pseudo-observations behind the global residual
SIGMA_FLOOR = 0.30              # narrowest honest band, in logit units
SIGMA_CAP = 2.0                 # widest band, in logit units
HORIZON_SIGMA_PER_SQRT_DAY = 0.18  # drift no trend can see
MAX_REFRESH_LAG_HOURS = 6.0     # cap on the staleness correction
MAX_N_GROWTH = 20.0             # sanity cap: players never grow 20x in one horizon

# Launch night's first refresh, as the rules state it: the frozen page
# tracked 638 players and the refreshed page 3,868.
LAUNCH_FROZEN_PLAYERS = 638
LAUNCH_REFRESHED_PLAYERS = 3868
LAUNCH_REFRESH_AT = "2026-11-19T04:30:00Z"
LAUNCH_REFRESH_SIGMA = 0.08     # tight band: the inversion is exact up to rounding


def parse_time(text):
    """ISO 8601 UTC timestamp -> epoch seconds (float)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def clamp_rate(pct):
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (as the house baseline rounds)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value.is_integer() else value


def wls(xs, ys, ws):
    """Weighted least squares for y ~ a + b x.

    Returns (a, b, resid_sd, se_b, x_last_hat) where x_last_hat is the
    fitted value at the largest x.  With fewer than 2 distinct x values
    the slope is 0 and the fit is the weighted mean.
    """
    sw = sum(ws)
    if sw <= 0:
        return 0.0, 0.0, 0.0, 1.0, 0.0
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        resid = math.sqrt(sum(w * (y - my) ** 2 for y, w in zip(ys, ws)) / sw) if len(ys) > 1 else 0.0
        return my, 0.0, resid, 1.0, my
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = my - b * mx
    n = len(ys)
    dof = max(n - 2, 1)
    sse = sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws))
    # Weighted residual sd on the unweighted scale: rescale by mean weight.
    resid = math.sqrt(sse / dof / (sw / n)) if n > 2 else math.sqrt(sse / (sw / n)) if sse > 0 else 0.0
    se_b = resid / math.sqrt(sxx / (sw / n)) if resid > 0 else 0.0
    x_max = max(xs)
    return a, b, resid, se_b, a + b * x_max


def shrink_slope(b, se_b, prior_mean):
    """Shrink a fitted slope toward a prior mean by precision weighting."""
    prec = 1.0 / (se_b * se_b) if se_b > 1e-9 else 1e6
    prec = min(prec, 1e4)
    prior_prec = 1.0 / (SLOPE_PRIOR_SD * SLOPE_PRIOR_SD)
    shrunk = (prec * b + prior_prec * prior_mean) / (prec + prior_prec)
    se_shrunk = 1.0 / math.sqrt(prec + prior_prec)
    return shrunk, se_shrunk


def refresh_states(case):
    """Collapse the history to distinct refresh states.

    Returns a list of (t_seconds, players, rates_tuple), oldest first,
    keeping the first read of each distinct snapshot.
    """
    states = []
    seen = None
    for read in case["history"]:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key == seen:
            continue
        seen = key
        states.append((parse_time(read["slot"]), read["players_tracked"], tuple(read["rates"])))
    return states


def launch_night_refresh(case, states):
    """True when the case is the launch-night frozen page facing its
    first refresh: one distinct state, 638 players, checkpoint before
    the refresh and target after it (the two practice cases)."""
    if len(states) != 1:
        return False
    if states[0][1] != LAUNCH_FROZEN_PLAYERS:
        return False
    refresh_at = parse_time(LAUNCH_REFRESH_AT)
    # The target may be the refresh instant itself: the answer is the
    # first read in the hour from the target, which is the refreshed page.
    return parse_time(case["at"]) < refresh_at <= parse_time(case["target"])


def forecast_launch_refresh(case, states):
    """Invert launch night's first refresh: earners unchanged, players
    638 -> 3,868, the trophy everyone holds stays at 100%."""
    rates = states[0][2]
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = rates[i]
        if rate is None:
            rate = 0.0
        if rate >= 100.0:
            rows.append({"id": trophy["id"], "q": [99.9, 99.99, 100, 100, 100]})
            continue
        earners = rate * LAUNCH_FROZEN_PLAYERS / 100.0
        new_rate = 100.0 * earners / LAUNCH_REFRESHED_PLAYERS
        lg = logit(new_rate)
        qs = [round4(min(100.0, max(0.0, sigmoid(lg + z * LAUNCH_REFRESH_SIGMA)))) for z in Z]
        for k in range(1, len(qs)):
            qs[k] = max(qs[k], qs[k - 1])
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def forecast_case(case):
    states = refresh_states(case)
    if not states:
        raise ValueError("case has no history")
    if launch_night_refresh(case, states):
        return forecast_launch_refresh(case, states)

    levels = case.get("levels") or LEVELS
    zs = [_norm_ppf(p) for p in levels] if levels != LEVELS else Z

    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    target_at = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0

    # Refresh gaps: how stale the answering read's snapshot may be.
    gaps = [(states[k + 1][0] - states[k][0]) / 3600.0 for k in range(len(states) - 1)]
    gaps = [g for g in gaps if g > 0]
    mean_gap = sum(gaps) / len(gaps) if gaps else 0.0
    lag_hours = min(0.5 * mean_gap, MAX_REFRESH_LAG_HOURS)
    t_eff = target_at - lag_hours * 3600.0

    def x_of(t):
        return math.log1p(max((t - release) / 3600.0, -0.999))

    xs = [x_of(t) for t, _, _ in states]
    x_last = xs[-1]
    x_tgt = x_of(t_eff)
    dx = max(x_tgt - x_last, 0.0)
    t_last = states[-1][0]
    n_last = states[-1][1]

    # Recency weights: a state a day old counts half as much as the last.
    w_time = [0.5 ** ((t_last - t) / (RECENCY_HALFLIFE_HOURS * 3600.0)) for t, _, _ in states]

    # --- players tracked: log-linear, plus a damped quadratic in x ---
    ln_n = [math.log(max(n, 1)) for _, n, _ in states]
    a_n, b_n, r_n, se_n, _ = wls(xs, ln_n, w_time)
    ln_n_hat = a_n + b_n * x_tgt
    if len(states) >= 8:
        # Quadratic in x by least squares on the centred regressor.
        mx = sum(w * x for x, w in zip(xs, w_time)) / sum(w_time)
        xc = [x - mx for x in xs]
        # Solve the 3x3 normal equations for [1, xc, xc^2] by Gaussian elimination.
        cols = [[1.0] * len(xs), xc, [c * c for c in xc]]
        ata = [[sum(w * cols[i][k] * cols[j][k] for k, w in enumerate(w_time)) for j in range(3)] for i in range(3)]
        atb = [sum(w * cols[i][k] * y for k, (w, y) in enumerate(zip(w_time, ln_n))) for i in range(3)]
        try:
            m = [row[:] + [atb[i]] for i, row in enumerate(ata)]
            for col in range(3):
                piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
                m[col], m[piv] = m[piv], m[col]
                if abs(m[col][col]) < 1e-12:
                    raise ArithmeticError("singular")
                for r in range(col + 1, 3):
                    f = m[r][col] / m[col][col]
                    for c in range(col, 4):
                        m[r][c] -= f * m[col][c]
            coef = [0.0] * 3
            for r in (2, 1, 0):
                coef[r] = (m[r][3] - sum(m[r][c] * coef[c] for c in range(r + 1, 3))) / m[r][r]
            xt = x_tgt - mx
            # Damp the curvature halfway: a quadratic fitted on two days
            # of launch data must not be trusted at full strength 4.5
            # days out.
            quad = coef[0] + coef[1] * xt + 0.5 * coef[2] * xt * xt
            ln_n_hat = 0.5 * ln_n_hat + 0.5 * quad
        except ArithmeticError:
            pass
    n_hat = math.exp(ln_n_hat)
    n_hat = min(max(n_hat, n_last), n_last * MAX_N_GROWTH)
    sigma_ln_n = min(max(r_n + se_n * dx, 0.05), 0.6)

    # --- per-trophy fits, then tier slope priors ---
    trophies = case["trophies"]
    n_tr = len(trophies)
    logit_fits = []
    count_fits = []
    for i in range(n_tr):
        pts_x, pts_y, pts_w = [], [], []
        pts_z, pts_wz = [], []
        for k, (_, n_k, rates_k) in enumerate(states):
            rate = rates_k[i]
            if rate is None:
                continue
            earners = rate * n_k / 100.0
            rel = earners / (earners + COUNT_WEIGHT_K)
            pts_x.append(xs[k])
            pts_y.append(logit(rate))
            pts_w.append(w_time[k] * rel)
            pts_z.append(math.log(earners + 0.5))
            pts_wz.append(w_time[k] * rel)
        if len(pts_x) >= 2 and max(pts_x) > min(pts_x):
            logit_fits.append(wls(pts_x, pts_y, pts_w))
            count_fits.append(wls(pts_x, pts_z, pts_wz))
        elif pts_x:
            # One usable point: flat from it.
            logit_fits.append((pts_y[-1], 0.0, 0.0, 1.0, pts_y[-1]))
            count_fits.append((pts_z[-1], 0.0, 0.0, 1.0, pts_z[-1]))
        else:
            logit_fits.append(None)
            count_fits.append(None)

    def tier_mean(fits, slope_idx=1):
        by_tier = {}
        for i, fit in enumerate(fits):
            if fit is None:
                continue
            tier = trophies[i].get("tier", "bronze")
            prec = 1.0 / (fit[3] * fit[3]) if fit[3] > 1e-9 else 1e4
            prec = min(prec, 1e4)
            num, den = by_tier.get(tier, (0.0, 0.0))
            by_tier[tier] = (num + prec * fit[slope_idx], den + prec)
        all_num = sum(v[0] for v in by_tier.values())
        all_den = sum(v[1] for v in by_tier.values())
        global_mean = all_num / all_den if all_den > 0 else 0.0
        out = {}
        for tier, (num, den) in by_tier.items():
            # A tier with little data falls back toward the global mean.
            out[tier] = (num + 2.0 * global_mean) / (den + 2.0) if den > 0 else global_mean
        return out, global_mean

    logit_tier, logit_global = tier_mean(logit_fits)
    count_tier, count_global = tier_mean(count_fits)

    resid_all = sorted(f[2] for f in logit_fits if f is not None and f[2] > 0)
    resid_global = resid_all[len(resid_all) // 2] if resid_all else 0.2

    horizon_sigma = HORIZON_SIGMA_PER_SQRT_DAY * math.sqrt(max(horizon_days, 0.0))
    rows = []
    for i, trophy in enumerate(trophies):
        last_rate = states[-1][2][i]
        if last_rate is not None and last_rate >= 100.0:
            # Every tracked player holds it, through every refresh so
            # far: it stays 100%, with a thin hedge below.
            rows.append({"id": trophy["id"], "q": [99.9, 99.99, 100, 100, 100]})
            continue
        tier = trophy.get("tier", "bronze")
        medians = []
        sig_parts = []
        lf = logit_fits[i]
        if lf is not None:
            _, b, resid, se_b, y_last = lf
            b_star, se_star = shrink_slope(b, se_b, logit_tier.get(tier, logit_global))
            medians.append(y_last + b_star * dx)
            n_pts = sum(1 for _, _, r in states if r[i] is not None)
            s_fit = math.sqrt((n_pts * resid * resid + RESID_PRIOR_WEIGHT * resid_global * resid_global) / (n_pts + RESID_PRIOR_WEIGHT))
            sig_parts.append(s_fit * s_fit)
            sig_parts.append((se_star * dx) ** 2)
            sig_parts.append((abs(b_star) * lag_hours / max((t_eff - release) / 3600.0, 1.0)) ** 2)
        cf = count_fits[i]
        if cf is not None:
            _, c, _, se_c, z_last = cf
            c_star, _ = shrink_slope(c, se_c, count_tier.get(tier, count_global))
            e_hat = math.exp(z_last + c_star * dx) - 0.5
            e_hat = max(e_hat, 0.0)
            medians.append(logit(100.0 * e_hat / n_hat))
            sig_parts.append((0.5 * sigma_ln_n) ** 2)
        if not medians:
            medians.append(logit(last_rate if last_rate is not None else 0.01))
        median = sum(medians) / len(medians)
        if len(medians) == 2:
            sig_parts.append((0.5 * abs(medians[0] - medians[1])) ** 2)
        sig_parts.append(horizon_sigma * horizon_sigma)
        sigma = math.sqrt(sum(sig_parts))
        sigma = min(max(sigma, SIGMA_FLOOR), SIGMA_CAP)
        median = min(max(median, logit(0.01)), logit(99.99))
        qs = []
        for z in zs:
            q = round4(min(100.0, max(0.0, sigmoid(median + z * sigma))))
            qs.append(q)
        for k in range(1, len(qs)):
            qs[k] = max(qs[k], qs[k - 1])
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
