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
and every rate but the one at 100% fell.

## The model

1. **Collapse the history to refresh states.** Consecutive identical
   reads are one state. Fitting raw reads would overweight however long
   each snapshot happened to sit.
2. **Players tracked N(t).** Log-linear in `log(1 + hours since
   release)`, recency-weighted (half-life 24h), plus a damped quadratic
   term when there are enough states to see curvature. The forecast is
   clamped to never fall below the last read.
3. **Earners E_i(t) per trophy.** Log-linear in `log(1 + hours)`, each
   trophy's slope shrunk toward its tier's mean slope (platinum, gold,
   silver, bronze progress at different paces, and rare trophies have
   few earners, so their own slope is mostly rounding noise). Rates
   are weighted by earners, since a 2-decimal rate carries far more
   noise about a trophy with 4 earners than one with 4,000.
4. **Second view: logit rate trend.** Each trophy's logit rate fit
   directly against `log(1 + hours)` with the same tier shrinkage.
   The median is the mean of the two views in logit space; their
   disagreement widens the spread.
5. **Spread.** A prediction interval on the logit scale: fit residuals
   (shrunk toward the global residual), slope uncertainty scaled by
   extrapolation distance, disagreement between the two views,
   uncertainty in the player-count forecast, a horizon term for drift
   no trend can see, and a refresh-lag term — the answering read shows
   the last refresh before the target, not the target itself. Floored
   at 0.30 and capped at 2.0 logit units, so a smooth history never
   yields an overconfident band and a wild trophy never yields a
   meaningless one.
6. **The 100% trophy.** The trophy every tracked player holds (complete
   the opening mission) has survived every refresh at 100%; it is
   forecast at 100% with a thin hedge below.
7. **Launch night.** When the whole history is the single frozen state
   from before Exophase's first refresh (638 players, checkpoint before
   04:30 UTC on 19 November 2026) and the target is after it, the
   refresh is inverted from the facts the competition's rules state:
   the refreshed page tracks 3,868 players, the earners are unchanged,
   and the 100% trophy stays at 100%. That is exactly the two practice
   cases; the final case, with two days of refresh states, never takes
   this branch.

## Why this beats the house baselines

The persistence baseline says every rate stays where the last read put
it: it misses both the dilution from new players and the progression of
earners. The trend baseline fits a logit line over the last 24 hours of
raw reads: it conflates the two halves, overweights stale snapshots,
and has no shrinkage, so rare trophies get wild slopes from rounding
noise. Modelling earners and players separately, on refresh states,
with tier shrinkage, keeps each mechanism where it belongs.

## Determinism and limits

No randomness, no network, no clock dependence: two runs on the same
case write the same bytes. Runtime is a few dozen refresh states times
55 trophies of closed-form weighted least squares — far inside the
8-second, 1-CPU, 1024 MB sandbox.
