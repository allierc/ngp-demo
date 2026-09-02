# A median filter in time, against the fit

The objection this answers: *the NGP movie shows the flicker leaving the data
and landing in the residual, but a running median over time would do the same
for nothing.*  It is right in half the cases, and the half is decidable.

Everything below is produced by two scripts in this repo and nothing else:

    python scripts/median_baseline.py --field bisons --seconds 8 \
        --modes stripe:0.1 stripe:0.2 stripe:0.5 stripe:0.8 shot:0.02 gauss skew \
        --windows 3 7 15 --ngp bisons_px1
    bash scripts/submit_sweep_inject.sh          # 48 L4 jobs, 20 s each
    python scripts/sweep_inject.py --table


## The data, and why not zapbench

The truth is `video_bisons.tif`, 257 frames of 100x100 — a natural video with no
artefact of its own — written into `data/bisons/field.zarr` in the layout the
pages already read.  Artefacts are then added on purpose, so everything in a
residual was put there by us and "recovering the truth" is a well-posed thing to
score.

The zapbench recording cannot play this role.  It already contains the stripes
under study, so its "truth" is itself striped and every number is ambiguous:
a method that reproduces the recording faithfully scores well precisely by
reproducing the artefact.  The zapbench numbers in the git history were measured
before this was fixed and should be read as no-ground-truth diagnostics only.

Artefact amplitude throughout: **0.2461**, which is 2.5x the frame-to-frame
standard deviation of the clip and 12.6% of its value range.


## Which regimes were tested

Four artefacts, chosen because they separate the two methods along two axes —
whether the corruption is a minority of a time window, and whether it is
symmetric:

| mode | what it is | sparse or dense | symmetric |
|---|---|---|---|
| `stripe:p` | one-sided row stripes, redrawn every frame, hitting a pixel in a fraction `p` of them | tunable by `p` | no |
| `shot:r` | isolated bright pixels at rate `r`, textbook impulse noise | sparse | no |
| `gauss` | zero-mean noise on every pixel of every frame | dense | yes |
| `skew` | exponential noise, same variance as `gauss` | dense | no |

The 48 sweep jobs used three of them: **`shot:0.02`** and **`stripe:0.1`**, the
two regimes where the median wins and tuning had something to prove, and
**`stripe:0.8`**, the one where the fit wins.


## The map: where each method belongs

Error of doing nothing divided by error of the method, so above 1.00 it helped
and below 1.00 it made the data worse.

| artefact | best median | ngp fit | winner |
|---|---|---|---|
| `shot:0.02` sparse impulses | **1.24x** (k=3) | 0.66x | median, by a lot |
| `stripe:0.1` sparse stripes | **1.51x** (k=3) | 1.19x | median |
| `stripe:0.2` | **1.32x** (k=3) | 1.24x | median, narrowly |
| `stripe:0.5` half the frames | 1.04x | **1.18x** | fit |
| `stripe:0.8` most frames | 0.98x | **1.05x** | fit; every median hurts |
| `gauss` dense, symmetric | 1.72x (k=7) | **2.42x** | fit, by a lot |
| `skew` dense, one-sided | **1.46x** (k=7) | 1.31x | median |

One mechanism explains the whole table.  **A median rejects outliers**, so it
wins while the artefact is a minority of its window and fails when it is the
majority — at duty 0.8 every window is corrupted, the median adopts the stripe
as its baseline, and all three window lengths end up worse than leaving the data
alone.  **A least-squares fit estimates a mean**, so it wins where nothing is an
outlier and there is nothing to reject.

The `skew` row is the honest counterexample to a claim made earlier in this
work.  On the recording the medians showed a baseline shift growing with window
(-0.125 to -0.280 per thousand of span for k=3/7/15) against -0.028 for the fit,
which I put as the fit being unbiased.  It is unbiased **for zero-mean noise**.
Noise with a non-zero mean shifts a least-squares fit by the whole of that mean
(0.2462 of an injected 0.2461) while the median shifts by only ln2 of it
(0.1998).  Neither removes an offset it was not told about.


## The same map with a MEAN filter instead

A running mean over the same window, same edge handling, so the only difference
between the two filter rows is the statistic.  This is the sharper comparison of
the two: a least-squares fit *estimates a mean*, so wherever the fit beats the
median, this table says whether it won by being a mean or by being a fit.

| artefact | best mean | best median | ngp fit | winner |
|---|---|---|---|---|
| `shot:0.02` sparse impulses | 0.88x (k=3) | **1.24x** (k=3) | 0.75x | median; the mean is worse than nothing |
| `stripe:0.1` sparse stripes | 1.31x (k=3) | **1.51x** (k=3) | 1.19x | median |
| `stripe:0.2` | **1.34x** (k=3) | 1.32x (k=3) | 1.27x | mean, by a hair |
| `stripe:0.5` half the frames | **1.19x** (k=3) | 1.04x (k=3) | 1.18x | mean and fit, level |
| `stripe:0.8` most frames | **1.06x** (k=3) | 0.98x (k=3) | 1.05x | mean and fit, level |
| `gauss` dense, symmetric | 1.90x (k=7) | 1.72x (k=7) | **2.07x** | fit |
| `skew` dense, one-sided | 1.25x (k=7) | **1.46x** (k=7) | 1.27x | median |

THE MEAN AND THE FIT TRACK EACH OTHER, and that is the finding.  On the three
regimes where the fit beat the median -- stripe 0.5, stripe 0.8 and gauss -- a
plain box filter beats it too, and by almost exactly the same margin (1.19 vs
1.18, 1.06 vs 1.05, 1.90 vs 2.07).  So the fit was not winning because it is a
fit; it was winning because it estimates a mean and the median does not, and on
those artefacts a mean is the right estimator.  The one place the fit is
genuinely ahead of both filters is `gauss`, where it has something a box filter
cannot have: it also averages over SPACE, at whatever scale each level works at.

Where the median wins it beats both -- `shot:0.02` is the clearest, since a
single bright pixel moves a 3-frame mean by a third of its amplitude and moves
the median not at all, which is why the mean there is worse than doing nothing
(0.88x) while the median is the best method in the table.

None of this changes what the fit is FOR.  A filter of either kind returns one
value per voxel and stores exactly what it was given; the fit stores 5.8 M
parameters for 2.57 M voxels here and, at the sizes that matter, far fewer than
the data -- and it answers at any (x, y, t) rather than only on the grid.  The
denoising comparison is a fair fight the fit does not need to win.

## The two regimes head to head

One NGP — L=16, T=2^22, 1 px and 2 frames per finest cell, 5.8 M parameters —
against the three windows.  The two RMSE columns split the same error by where
it happened.

**Sporadic stripes, 10% of frames per pixel**

| method | RMSE vs truth | RMSE on stripes | RMSE where clean | vs doing nothing |
|---|---|---|---|---|
| doing nothing | 0.0778 | 0.2461 | 0 | — |
| **median k=3** | **0.0517** | 0.1227 | 0.0360 | **1.51x** |
| median k=7 | 0.0814 | 0.1057 | 0.0782 | 0.96x |
| median k=15 | 0.1255 | 0.1370 | 0.1242 | 0.62x |
| ngp fit | 0.0656 | 0.1302 | 0.0538 | 1.19x |

**Frequent stripes, 80% of frames per pixel**

| method | RMSE vs truth | RMSE on stripes | RMSE where clean | vs doing nothing |
|---|---|---|---|---|
| doing nothing | 0.2201 | 0.2461 | 0 | — |
| median k=3 | 0.2255 | 0.2371 | 0.1710 | 0.98x |
| median k=7 | 0.2359 | 0.2406 | 0.2162 | 0.93x |
| median k=15 | 0.2531 | 0.2550 | 0.2451 | 0.87x |
| **ngp fit** | **0.2098** | 0.2256 | **0.1284** | **1.05x** |

A longer window always removes more stripe and always costs more elsewhere: in
the sporadic regime the stripe residue falls from 0.1227 at k=3 towards 0.1057
at k=7 while the clean pixels degrade 0.0360 to 0.0782, which is why k=3 wins
and k=15 is worse than doing nothing.  In the frequent regime no window removes
the stripe at all — the residue sticks at 0.24 of the 0.2461 injected — so the
medians pay the collateral cost for nothing.  The fit is the only method above
1.00x there, and it is much the cleanest where the artefact never landed.

Movies, 8 s each at 32.125 fps, in `out/`:
`bisons_stripe0.1_montage`, `bisons_stripe0.8_montage` (truth / corrupted / best
median / fit) and the two `_montage_err` versions, whose 2x2 reads across the
top — the injected artefact top-left, the fit top-right — on one shared signed
scale taken from the 99.9th percentile of every method's error.


## What the 48 sweep jobs taught

16 settings, four knobs swept one at a time about L=16, T=2^22, 2 px, 2 frames,
each run against all three regimes: 48 jobs, one per L4, 20 s per fit.

**1. The hypothesis was half right.**  A stripe one row high and one frame long
is representable only by the finest levels, so coarsening the ladder should
force the fit to average it away.  It does: on `shot:0.02` the error left where
the artefact was falls fivefold as the finest cell grows from 2 px to 8 px.  But
the *total* error rises, 0.052 to 0.087, because the same coarsening destroys
more signal than artefact.  Rejecting an artefact and reconstructing a scene are
in direct competition, and on these clips the scene wins.

**2. So the best setting was the finest one tried, in all three regimes** — 1 px
per finest cell, 3.45x the parameters — worth 12.1% on `shot:0.02`, 4.5% on
`stripe:0.1` and 0.7% on `stripe:0.8`.  Not the coarse, capacity-starved setting
the artefact argument predicted.

**3. Tuning changes no ranking.**  Where the median wins it still wins against
the best fit found (k=3 scores 0.028 on `shot:0.02` against the tuned fit's
0.046), and where the fit wins it wins with the settings it already had.  The
gap between the two methods is a difference in what they estimate, not a
difference a hyperparameter can close.

**4. The knobs are not interchangeable.**  Table size is nearly inert here —
2^18 through 2^23 span 0.050 to 0.052 on `shot:0.02`, because a 100x100x257
volume never fills a 2^22 table — while the finest cell size moves the same
number from 0.046 to 0.087.  On this data the ladder's *resolution* is the knob
and its *capacity* is not; on zapbench, 40x larger, the reverse is true.

Raw results: `log/sweep_inject/*.json`, one per job.
