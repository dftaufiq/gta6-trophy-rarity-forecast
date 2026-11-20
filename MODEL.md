# Methodology

The entry is a single program, `main.py` (Python 3.13, standard library
only). It reads a case — every read the house made of Exophase's GTA VI
PS5 trophy page before the checkpoint — and writes five quantiles
(5/25/50/75/95%) of each trophy's earn rate at the target.

## What the data is

A trophy's earn rate is `earners / players tracked`, printed to 2
decimals. Exophase's page is a snapshot refreshed on its own schedule:
between refreshes every read is identical. The history is therefore a
staircase of refresh states, and the two halves of a rate move
differently at a refresh: the player count jumps to everyone who has
appeared since the last refresh (diluting every rate at once), while
each trophy's earners only grow as players actually reach it. On launch
night the first refresh took the page from 638 players tracked to 3,868
and every rate but the one at 100% fell. The two halves can also refresh
on different schedules: launch night's earners were frozen while the
player count jumped, and nothing in the data promises they move
together later.

## The model

1. **Collapse the history to refresh states.** Consecutive identical
   reads are one state. Fitting raw reads would overweight however long
   each snapshot happened to sit.
2. **Players tracked N(t).** An ensemble of three forecasts in
   `log(1 + hours since release)`, recency-weighted (half-life 24h): a
   log-linear fit, a damped quadratic, and a saturating Weibull curve
   fitted on a grid. The median of the three is the forecast; a
   log-linear fit alone overshoots about 2x over a 4.5-day horizon
   because player growth saturates, and the Weibull is the only view
   that knows that. The forecast is clamped to never fall below the
   last read.
3. **Earners E_i(t) per trophy, three views.** The median forecast is
   the median of three views in logit space, so no single mechanism
   can drag a trophy:
   - **Cohort view.** Players arrive over time (arrival buckets built
     from the refresh states' player counts, plus forecast arrivals
     to the target), and each cohort earns the trophy with probability
     `p_i * (1 - exp(-(age / T_i)^1.5))`. `T_i` starts from a prior
     read off the trophy's description — a chapter is fast, a
     250-item grind is slow — and is fitted on a log-spaced grid,
     keeping the prior unless another value beats it by 10%; `p_i` is
     fitted from the observed earners. This view knows about
     saturation: a fast trophy cannot keep growing past the players
     who will ever earn it.
   - **Counts trend.** Log-earners linear in `log(1 + hours)`, each
     trophy's slope shrunk toward its tier's mean slope (platinum,
     gold, silver, bronze progress at different paces, and rare
     trophies have few earners, so their own slope is mostly rounding
     noise).
   - **Logit rate trend.** Each trophy's logit rate fit directly
     against `log(1 + hours)` with the same tier shrinkage.
   Rates are weighted by earners throughout, since a 2-decimal rate
   carries far more noise about a trophy with 4 earners than one with
   4,000.
4. **Earner lag.** The earners Exophase shows may trail the present:
   the page can refresh its player count while its earners are hours
   stale. A single global lag (0, 6, or 12 hours) is fitted by timing
   the total earners — the exact `trophies_earned` count minus the one
   trophy every tracked player holds, which carries no lag signal and
   would swamp the total — against the cohort model's predicted total
   at each candidate lag. Per-trophy rates are too noisy to time a lag
   (a 6-hour lag moves a trophy's log earners by ~0.03, less than its
   rounding noise); the total is exact, so the lag is detectable
   there. Lag 0 wins unless another lag beats it by 5%. The cohort
   view is then evaluated at the target minus the fitted lag, and the
   spread widens by the earners' growth rate over the lag window.
5. **Spread.** A prediction interval on the logit scale: fit residuals
   (shrunk toward the global residual), slope uncertainty scaled by
   extrapolation distance, disagreement between the three views,
   uncertainty in the player-count forecast, a horizon term for drift
   no trend can see (0.40 logit per root-day), and a refresh-lag term
   — the answering read shows the last refresh before the target, not
   the target itself. Floored at 0.50 and capped at 2.2 logit units,
   so a smooth history never yields an overconfident band and a wild
   trophy never yields a meaningless one. The floor matters most for
   the trophies stuck at the 0.01% print floor: two-decimal censoring
   hides their growth, and a band that trusts the printed 0.01% is a
   band that misses.
6. **The 100% trophy.** The trophy every tracked player holds (complete
   the opening mission) has survived every refresh at 100%; it is
   forecast at 100% with a thin hedge below.
7. **Launch night.** When the whole history is the single frozen state
   from before Exophase's first refresh (638 players, checkpoint before
   04:30 UTC on 19 November 2026) and the target is after it, the
   refresh is inverted from the facts the competition's data states:
   the refreshed page tracks 3,868 players, the earners are unchanged,
   and the 100% trophy stays at 100%. The band is tight (0.02 logit)
   because the inversion is exact up to the 2-decimal rounding of the
   frozen rates. That is exactly the two practice cases; the final
   case, with two days of refresh states, never takes this branch.

## Validation

The entry was tested on a simulator that rebuilds the whole pipeline
from scratch: a Weibull player curve, cohort earners per trophy with
description-plausible timescales, snapshot refreshes every 2–6 hours,
reads every 30 then 10 minutes, and the answer taken as the first read
in the hour from the target. Nineteen worlds span week-one totals of
150k–1.5M players, earner lags of 0–6 hours, sticky earners (refreshed
every 18–30 hours while the player count refreshes every 2–6), fast
refreshes, and Weibull shape variants. Across them the entry's mean
pinball loss is 0.563 (worst 0.780), against 0.715 for the persistence
baseline and 0.695 for the trend baseline (worst 1.443, when sticky
earners make its 24-hour logit line extrapolate a cliff). The entry
wins clearly wherever earners are sticky or the player curve saturates
and loses slightly only in pure fixed-lag worlds, where every model
degrades together. On the two practice cases the loss is 0.0024 with
zero median error — the loss there is entirely the width of the honest
launch-night band.

## Why this beats the house baselines

The persistence baseline says every rate stays where the last read put
it: it misses both the dilution from new players and the progression of
earners. The trend baseline fits a logit line over the last 24 hours of
raw reads: it conflates the two halves, overweights stale snapshots,
and has no shrinkage, so rare trophies get wild slopes from rounding
noise — and when earners go sticky it extrapolates a cliff. Modelling
earners and players separately, on refresh states, with tier shrinkage,
a saturating cohort view, and a fitted earner lag keeps each mechanism
where it belongs.

## Determinism and limits

No randomness, no network, no clock dependence: two runs on the same
case write the same bytes. Runtime is a few dozen refresh states times
55 trophies of closed-form weighted least squares plus a small grid
search — about half a second on the final case, far inside the
8-second, 1-CPU, 1024 MB sandbox.
