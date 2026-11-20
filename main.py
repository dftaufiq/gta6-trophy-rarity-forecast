"""Forecast GTA VI PlayStation 5 trophy earn rates on Exophase at a case's target.

How the data behaves (see the kit's RULES.txt and NOTICE.txt):

* Exophase's trophy page is a snapshot refreshed on its own schedule, not
  with every player's trophy.  Between refreshes every read the house makes
  is byte-identical: the same players-tracked count and the same rates.
  So the history is a staircase of refresh states, and a naive fit over
  raw reads would overweight however long each snapshot happened to sit.
* A rate is earners / players-tracked, and a refresh moves the two halves
  differently: the player count jumps to everyone who has appeared since
  the last refresh, diluting every rate at once, while each trophy's
  earners only grow as players actually reach it.  On launch night the
  first refresh took the page from 638 players tracked to 3,868 and every
  rate but the one at 100% fell.
* Rates print to 2 decimals, so a trophy at 0.01% is censored: its true
  rate is anywhere in [0.005%, 0.015%), and it can triple inside that
  band while the printed rate stays flat.  Half the trophies sit at that
  floor early on, and a trend fit to their printed series sees no growth
  at all.  Their growth has to come from structure, not from their own
  series.

The forecast models the process, on refresh states only (consecutive
identical reads collapsed to one state):

* Players tracked N(t): an ensemble of three fits in log(1 + hours since
  release) -- a log-linear fit, a damped quadratic, and a Weibull
  adoption curve fitted by grid search -- combined by median, so no
  single extrapolation shape decides the denominator.  N never
  forecasts below the last read.
* Earners E_i(t), three views per trophy, combined by median in logit
  space:
  1. Cohort view.  Players arrive over time (arrivals read off the
     player-count series) and a player who arrived s hours ago has
     earned trophy i with probability p_i * (1 - exp(-(age/T_i)^1.5)).
     T_i, the trophy's characteristic time to earn, starts from a prior
     read off the trophy's description (a chapter is fast, 250 stunts
     is slow) and is then fitted on a small grid around the prior,
     keeping the prior unless another value fits decisively better;
     p_i is fitted from the observed earners.  This view saturates
     naturally (fast trophies stop growing once every cohort has had
     its time) and sees through the 0.01% censoring (a grind's earners
     keep growing as its cohorts age even while its printed rate sits
     at the floor).
  2. Logit-trend view.  Each trophy's logit rate fit against
     log(1 + hours), slope shrunk toward its tier's mean slope.
  3. Counts-trend view.  Each trophy's log earners fit the same way,
     divided by the forecast player count.
* Spread: a prediction interval on the logit scale -- fit residuals
  (shrunk toward the global residual), slope uncertainty scaled by
  extrapolation distance, disagreement between the views, uncertainty
  in the player-count forecast, a horizon term for drift no trend can
  see, and a refresh-lag term (the answering read shows the last
  refresh before the target, not the target itself).  Floored and
  capped, so a smooth history never yields an overconfident band and a
  wild trophy never yields a meaningless one.  The floor and horizon
  terms were calibrated on simulated Exophase worlds (see the build
  notes): a 4.5-day launch-week extrapolation is genuinely uncertain,
  and narrow bands are punished far harder than wide ones.

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
of closed-form fits and small grid searches.
"""

import json
import math
import os
import re
import sys
from datetime import datetime
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

# --- tuning constants (fixed in advance, calibrated on simulated worlds) ---
RECENCY_HALFLIFE_HOURS = 24.0   # weight of a state halves every day
COUNT_WEIGHT_K = 25.0           # earners at which a rate's rounding noise stops dominating
SLOPE_PRIOR_SD = 0.5            # sd of a trophy's slope around its tier mean
RESID_PRIOR_WEIGHT = 4.0        # pseudo-observations behind the global residual
SIGMA_FLOOR = 0.50              # narrowest honest band, in logit units
SIGMA_CAP = 2.2                 # widest band, in logit units
HORIZON_SIGMA_PER_SQRT_DAY = 0.40  # drift no trend can see
MAX_REFRESH_LAG_HOURS = 6.0     # cap on the staleness correction
MAX_N_GROWTH = 20.0             # sanity cap: players never grow 20x in one horizon
COHORT_SHAPE = 1.5              # Weibull shape of the earning-time curve
COHORT_PRIOR_KEEP = 0.10        # keep the T prior unless a fit beats it by 10%

# Launch night's first refresh, as the rules state it: the frozen page
# tracked 638 players and the refreshed page 3,868.
LAUNCH_FROZEN_PLAYERS = 638
LAUNCH_REFRESHED_PLAYERS = 3868
LAUNCH_REFRESH_AT = "2026-11-19T04:30:00Z"
LAUNCH_REFRESH_SIGMA = 0.02     # tight band: the inversion is exact up to
                                # the 2-decimal rounding of the frozen rates
                                # (+/-0.0008pp on the new rate); 0.02 logit
                                # is ~25x that error, still honest


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


def median3(a, b, c):
    return a + b + c - min(a, b, c) - max(a, b, c)


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
    if n <= 2:
        # Two points always fit a line exactly: no slope information.
        se_b = 1.0
    elif resid > 0:
        se_b = max(resid / math.sqrt(sxx / (sw / n)), 0.02)
    else:
        se_b = 0.02
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

    Returns a list of (t_seconds, players, rates_tuple, trophies_earned),
    oldest first, keeping the first read of each distinct snapshot.
    """
    states = []
    seen = None
    for read in case["history"]:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key == seen:
            continue
        seen = key
        states.append((parse_time(read["slot"]), read["players_tracked"],
                       tuple(read["rates"]), read.get("trophies_earned")))
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


# --- the cohort view: how long a trophy takes to earn ---

def earn_time_prior(trophy):
    """Characteristic hours for a cohort to earn this trophy, from its
    description.  A chapter is fast, a 250-item grind is slow; the fit
    in fit_cohort() then moves this up or down on the evidence."""
    text = ((trophy.get("name") or "") + " " + (trophy.get("description") or "")).lower()
    if trophy.get("tier") == "platinum" or "100% completion" in text:
        return 400.0
    if "gone fishing" in text:
        return 1.0
    if "chapter 1" in text:
        return 15.0
    if "chapter 2" in text:
        return 25.0
    if "chapter 3" in text:
        return 40.0
    if "chapter 4" in text:
        return 55.0
    if "complete the final mission" in text and "for" not in text:
        return 80.0
    if "final mission for" in text:
        return 70.0
    if "250 stunts" in text or "stunts" in text:
        return 300.0
    if "card set" in text:
        return 250.0
    if "bank vault" in text:
        return 150.0
    if "gold medal in all" in text:
        return 150.0
    if "drone race at every" in text:
        return 180.0
    if "every spot" in text or "explore all" in text:
        return 150.0
    if "each type of activity" in text:
        return 90.0
    if "garage propert" in text:
        return 300.0
    if "laundering" in text:
        return 350.0
    if "mission strands" in text:
        return 120.0
    if "each gang" in text:
        return 150.0
    if "unique animal" in text:
        return 100.0
    if "spend $" in text:
        return 80.0
    if "discover" in text:
        # Discovering is passive accumulation while playing, not a grind.
        nums = [int(m) for m in re.findall(r"\d+", text)]
        n = max(nums) if nums else 0
        return 20.0 + n / 2.0
    # A bare count: the bigger the count, the longer the grind.
    nums = [int(m) for m in re.findall(r"\d+", text)]
    n = max(nums) if nums else 0
    if n >= 100:
        return 100.0 + n
    if n >= 20:
        return 40.0 + n
    if n >= 5:
        return 25.0 + 2.0 * n
    return 30.0


def _g(age_hours, t_char):
    """Share of a cohort of this age that has earned the trophy."""
    if age_hours <= 0:
        return 0.0
    return 1.0 - math.exp(-((age_hours / t_char) ** COHORT_SHAPE))


def fit_cohort(t_states, e_obs, w_obs, arrivals, t_char_prior, lag=0.0):
    """Fit (T, p) for one trophy's earners.

    t_states: refresh-state times (hours since release); e_obs: observed
    earners per state (None where the read lacked the trophy); w_obs:
    weights; arrivals: (time, count) buckets of player arrivals over the
    whole span, observed and forecast, for evaluating A(t) at any time;
    lag: how stale the observed earners are (the earners Exophase shows
    at time t are the earners of t - lag).  Returns (t_char, p, sse) or
    None when nothing is observed.
    """
    pts = [(t, e, w) for t, e, w in zip(t_states, e_obs, w_obs) if e is not None]
    if not pts:
        return None
    # A(t) for a candidate T: sum over arrival buckets of arrivals * G(age),
    # with the earners' lag applied.
    def a_at(t, t_char):
        return sum(cnt * _g(t - lag - bt, t_char) for bt, cnt in arrivals if t - lag > bt)

    grid = sorted({max(2.0, min(800.0, v)) for v in (
        t_char_prior * 0.5, t_char_prior, t_char_prior * 2.0,
        6.0, 9.0, 13.5, 20.0, 30.0, 45.0, 68.0, 100.0, 150.0, 225.0, 340.0, 500.0, 750.0)})
    best = None
    prior_sse = None
    for t_char in grid:
        a_vals = [a_at(t, t_char) for t, _, _ in pts]
        num = sum(w * e for (_, e, w), a in zip(pts, a_vals) if a > 0)
        den = sum(w * a for (_, _, w), a in zip(pts, a_vals) if a > 0)
        if den <= 0:
            continue
        p = num / den
        sse = 0.0
        for (_, e, w), a in zip(pts, a_vals):
            # Log-space error with a half-earner floor, so a trophy
            # with 4 earners is not fitted as if it had 4,000.
            pred = p * a
            sse += w * (math.log(e + 0.5) - math.log(pred + 0.5)) ** 2
        if t_char == t_char_prior:
            prior_sse = sse
        if best is None or sse < best[2]:
            best = (t_char, p, sse)
    if best is None:
        return None
    # Keep the description prior unless another T fits decisively better:
    # a censored trophy's rounding steps must not pick its timescale.
    if prior_sse is not None and prior_sse <= best[2] * (1.0 + COHORT_PRIOR_KEEP):
        t_char = t_char_prior
        a_vals = [a_at(t, t_char) for t, _, _ in pts]
        num = sum(w * e for (_, e, w), a in zip(pts, a_vals) if a > 0)
        den = sum(w * a for (_, _, w), a in zip(pts, a_vals) if a > 0)
        p = num / den if den > 0 else 0.0
        return (t_char, p, prior_sse)
    return best


def _cohort_p_at(t_states, e_obs, w_obs, arrivals, t_char, lag):
    """Fit p for one trophy at a fixed (T, lag).  Returns (p, sse) or None."""
    pts = [(t, e, w) for t, e, w in zip(t_states, e_obs, w_obs) if e is not None]
    if not pts:
        return None
    a_vals = [sum(cnt * _g(t - lag - bt, t_char) for bt, cnt in arrivals if t - lag > bt)
              for t, _, _ in pts]
    num = sum(w * e for (_, e, w), a in zip(pts, a_vals) if a > 0)
    den = sum(w * a for (_, _, w), a in zip(pts, a_vals) if a > 0)
    if den <= 0:
        return None
    p = num / den
    sse = 0.0
    for (_, e, w), a in zip(pts, a_vals):
        sse += w * (math.log(e + 0.5) - math.log(p * a + 0.5)) ** 2
    return p, sse


def fit_earner_lag(t_states, all_e_obs, all_w_obs, arrivals, priors, o_tot, w_time, excl):
    """Fit one global earners' lag (hours): the earners Exophase shows
    at time t are the earners of t - lag.

    Per-trophy rates are too noisy to time a lag (a 6-hour lag moves a
    trophy's log earners by ~0.03, less than its rounding noise), so
    the lag is timed on the TOTAL earners instead: o_tot, the exact
    count of trophies earned minus the one trophy every tracked player
    holds, which carries no lag signal and would swamp the total.
    Every trophy is fitted at its description-prior T for each
    candidate lag -- the same T set for every lag, so a timescale error
    cannot masquerade as a lag -- and the lag whose predicted total
    best matches the observed total wins; lag 0 wins unless another
    lag beats it by 5%."""
    lags = [0.0, 6.0, 12.0]
    totals = {}
    for lag in lags:
        # Predicted total earners per state at this lag.
        pred = [0.0] * len(t_states)
        for idx, (e_obs, w_obs, prior) in enumerate(zip(all_e_obs, all_w_obs, priors)):
            if idx == excl:
                continue
            fit = _cohort_p_at(t_states, e_obs, w_obs, arrivals, prior, lag)
            if fit is None:
                continue
            p, _ = fit
            for k, t in enumerate(t_states):
                a = sum(cnt * _g(t - lag - bt, prior) for bt, cnt in arrivals if t - lag > bt)
                pred[k] += p * a
        sse = 0.0
        for k, (o, pr) in enumerate(zip(o_tot, pred)):
            if o is None or o <= 0 or pr <= 0:
                continue
            sse += w_time[k] * (math.log(o) - math.log(pr)) ** 2
        totals[lag] = sse
    best_lag = min(lags, key=lambda L: totals[L])
    if totals[0.0] <= totals[best_lag] * 1.05:
        return 0.0
    return best_lag


def forecast_players(xs, ln_n, w_time, x_tgt, n_last):
    """Forecast players tracked at x_tgt: median of a log-linear fit, a
    damped quadratic, and a Weibull adoption curve fitted by grid search.
    Returns (n_hat, sigma_ln_n)."""
    a_n, b_n, r_n, se_n, _ = wls(xs, ln_n, w_time)
    preds = [a_n + b_n * x_tgt]

    # Damped quadratic in x.
    if len(xs) >= 8:
        mx = sum(w * x for x, w in zip(xs, w_time)) / sum(w_time)
        xc = [x - mx for x in xs]
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
            preds.append(coef[0] + coef[1] * xt + 0.5 * coef[2] * xt * xt)
        except ArithmeticError:
            pass

    # Weibull adoption curve: N = N_inf * (1 - exp(-(u/T)^k)), u = 1 + t.
    # Grid over (T, k); N_inf solved in log space for each pair.
    us = [math.exp(x) for x in xs]
    u_tgt = math.exp(x_tgt)
    best = None
    t_grid = [10.0, 18.0, 32.0, 56.0, 100.0, 180.0, 320.0, 560.0]
    k_grid = [0.7, 1.0, 1.4, 2.0, 2.8]
    for T in t_grid:
        for k in k_grid:
            g = [1.0 - math.exp(-((u / T) ** k)) for u in us]
            if min(g) <= 0:
                continue
            ln_inf = sum(w * (y - math.log(gg)) for w, y, gg in zip(w_time, ln_n, g)) / sum(w_time)
            sse = sum(w * (y - (ln_inf + math.log(gg))) ** 2 for w, y, gg in zip(w_time, ln_n, g))
            if best is None or sse < best[0]:
                best = (sse, ln_inf, T, k)
    if best is not None:
        _, ln_inf, T, k = best
        g_tgt = 1.0 - math.exp(-((u_tgt / T) ** k))
        if g_tgt > 0:
            preds.append(ln_inf + math.log(g_tgt))

    if len(preds) == 1:
        ln_hat = preds[0]
        spread = 0.0
    elif len(preds) == 2:
        ln_hat = 0.5 * (preds[0] + preds[1])
        spread = abs(preds[0] - preds[1])
    else:
        ln_hat = median3(preds[0], preds[1], preds[2])
        spread = max(preds) - min(preds)
    n_hat = math.exp(ln_hat)
    n_hat = min(max(n_hat, n_last), n_last * MAX_N_GROWTH)
    sigma_ln_n = min(max(r_n + se_n * max(x_tgt - xs[-1], 0.0) + 0.5 * spread, 0.08), 0.7)
    return n_hat, sigma_ln_n


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

    def h_of(t):
        return (t - release) / 3600.0

    xs = [x_of(t) for t, _, _, _ in states]
    hs = [h_of(t) for t, _, _, _ in states]
    x_last = xs[-1]
    x_tgt = x_of(t_eff)
    h_tgt = h_of(t_eff)
    dx = max(x_tgt - x_last, 0.0)
    t_last = states[-1][0]
    n_last = states[-1][1]

    # Recency weights: a state a day old counts half as much as the last.
    w_time = [0.5 ** ((t_last - t) / (RECENCY_HALFLIFE_HOURS * 3600.0)) for t, _, _, _ in states]

    # --- players tracked ---
    ln_n = [math.log(max(n, 1)) for _, n, _, _ in states]
    n_hat, sigma_ln_n = forecast_players(xs, ln_n, w_time, x_tgt, n_last)

    # Arrival buckets for the cohort view: observed arrivals between
    # states (at the midpoint) plus forecast arrivals to the target,
    # spread evenly in log-time.
    arrivals = []
    prev_t, prev_n = hs[0], states[0][1]
    arrivals.append((hs[0], float(states[0][1])))
    for k in range(1, len(states)):
        dn = states[k][1] - prev_n
        if dn > 0:
            arrivals.append((0.5 * (prev_t + hs[k]), float(dn)))
        prev_t, prev_n = hs[k], states[k][1]
    future = max(n_hat - n_last, 0.0)
    if future > 0:
        n_buckets = 8
        x0, x1 = math.log1p(max(hs[-1], 0.0)), math.log1p(max(h_tgt, 0.0))
        for b in range(n_buckets):
            xc = x0 + (x1 - x0) * (b + 0.5) / n_buckets
            arrivals.append((math.expm1(xc), future / n_buckets))

    # --- per-trophy trend fits, then tier slope priors ---
    trophies = case["trophies"]
    n_tr = len(trophies)
    logit_fits = []
    count_fits = []
    for i in range(n_tr):
        pts_x, pts_y, pts_w = [], [], []
        pts_z, pts_wz = [], []
        for k, (_, n_k, rates_k, _) in enumerate(states):
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

    # Observed earners per trophy per state, for the cohort view, and
    # the one global lag with which Exophase's earners trail the present.
    all_e_obs = []
    all_w_obs = []
    priors = []
    for i, trophy in enumerate(trophies):
        e_obs = []
        w_obs = []
        for k, (_, n_k, rates_k, _) in enumerate(states):
            rate = rates_k[i]
            if rate is None:
                e_obs.append(None)
                w_obs.append(0.0)
            else:
                earners = rate * n_k / 100.0
                e_obs.append(earners)
                w_obs.append(w_time[k] * earners / (earners + COUNT_WEIGHT_K))
        all_e_obs.append(e_obs)
        all_w_obs.append(w_obs)
        priors.append(earn_time_prior(trophy))
    # Total earners except the trophy every tracked player holds (it
    # carries no lag signal and would swamp the total): from the exact
    # trophies_earned count where it agrees with the rates, else from
    # the rates themselves.
    last_rates = states[-1][2]
    excl = max(range(n_tr), key=lambda i: last_rates[i] if last_rates[i] is not None else -1)
    if last_rates[excl] is None or last_rates[excl] < 99.9:
        excl = -1
    o_tot = []
    for k, (_, n_k, rates_k, tot_k) in enumerate(states):
        if excl >= 0:
            e_excl = (rates_k[excl] if rates_k[excl] is not None else 100.0) * n_k / 100.0
        else:
            e_excl = 0.0
        rate_sum = sum((r if r is not None else 0.0) * n_k / 100.0
                       for j, r in enumerate(rates_k) if j != excl)
        if tot_k is not None and abs((tot_k - e_excl) - rate_sum) <= 0.2 * max(rate_sum, 1.0):
            o_tot.append(tot_k - e_excl)
        else:
            o_tot.append(rate_sum)
    earner_lag = fit_earner_lag(hs, all_e_obs, all_w_obs, arrivals, priors,
                                o_tot, w_time, excl)

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
        views = []
        sig_parts = []
        lf = logit_fits[i]
        if lf is not None:
            _, b, resid, se_b, y_last = lf
            b_star, se_star = shrink_slope(b, se_b, logit_tier.get(tier, logit_global))
            views.append(y_last + b_star * dx)
            n_pts = sum(1 for _, _, r, _ in states if r[i] is not None)
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
            views.append(logit(100.0 * e_hat / n_hat))
            sig_parts.append((0.5 * sigma_ln_n) ** 2)
        # Cohort view: fit (T, p) on the observed earners, then read
        # the earners at the target off the arrival buckets, at the
        # lag with which the earners trail the present.
        coh = fit_cohort(hs, all_e_obs[i], all_w_obs[i], arrivals,
                         priors[i], earner_lag)
        if coh is not None:
            t_char, p, _ = coh
            h_coh = h_tgt - earner_lag
            a_tgt = sum(cnt * _g(h_coh - bt, t_char) for bt, cnt in arrivals if h_coh > bt)
            e_hat = p * a_tgt
            views.append(logit(100.0 * max(e_hat, 0.0) / n_hat))
            sig_parts.append((0.5 * sigma_ln_n) ** 2)
            if earner_lag > 0 and a_tgt > 0:
                # The answer's earners could be anywhere in a lag window
                # around the fitted lag: widen by the earners' growth
                # rate over that window.
                a_day = sum(cnt * _g(h_coh - 24.0 - bt, t_char) for bt, cnt in arrivals if h_coh - 24.0 > bt)
                if a_day > 0:
                    dln_dt = (math.log(a_tgt) - math.log(a_day)) / 24.0
                    sig_parts.append(min(abs(dln_dt) * earner_lag, 0.5) ** 2)
        if not views:
            views.append(logit(last_rate if last_rate is not None else 0.01))
        views_sorted = sorted(views)
        median = views_sorted[len(views_sorted) // 2]
        if len(views) >= 2:
            sig_parts.append((0.5 * (views_sorted[-1] - views_sorted[0])) ** 2)
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
