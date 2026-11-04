# Speaker-pair trajectory runs: results (fetched 2026-11-04)

Small outputs of the two speaker-pair runs on the cluster (`~/anon-bench/trajectory/<run>/`); the trajectories
themselves (26 GB + 6.9 GB of float16 velocities) stay there. Protocol and file format: `../README.md`.

| run | conversations | speakers | pairs | conversions | GPU batch | job | GPU time |
|---|---|---|---|---|---|---|---|
| `pairs_combined` (set 1) | 2190472 2215699 2392867 2508371 2577003 2751425 3017365 3228806 3229783 3303284 | 20 | 190 | 3,800 | 1 | 15285 (run), 15901 (MSE) | 36 min |
| `pairs_combined_2` (set 2) | 3334594 3340696 3385730 3413937 3433042 | 10 | 45 | 900 | 2 | 15892 | 58 min (shared node, slow storage) |

Per run: `index.csv` (one row per conversion), `selection.json` (speakers, segments, references), `select.log`,
`run.log` (per-conversion timing, GPU peak, batch check), `mse/mse_per_conversion.csv` (mse_00..mse_30 and `energy` =
mean of x_30^2), `mse/mse_summary.csv`, `mse/mse_by_noise.csv`, `mse/mse_vs_step.png`, `mse.log` (the printed table).

## MSE(x_k, x_30) per Euler step, mel space

MSE_k = mean over all 80 x frames mel bins of (x_k - x_30)^2, with x_k rebuilt from the saved velocities
(x_{k+1} = x_k + dt v_k; x_30 = the saved final mel, so MSE_30 = 5e-6 is float16 rounding only).

| step k | t | set 1 mean (3,800) | set 2 mean (900) | (1 - t)^2 |
|---|---|---|---|---|
| 0 | 0 | 46.05 | 54.05 | 1 |
| 5 | 0.17 | 31.61 | 37.06 | 0.694 |
| 10 | 0.33 | 20.18 | 23.67 | 0.444 |
| 15 | 0.50 | 11.34 | 13.30 | 0.250 |
| 20 | 0.67 | 5.04 | 5.91 | 0.111 |
| 25 | 0.83 | 1.26 | 1.48 | 0.028 |
| 29 | 0.97 | 0.050 | 0.059 | 0.0011 |
| 30 | 1 | 5e-6 | 6e-6 | 0 |

Findings (both sets, checked on `mse_per_conversion.csv`):

1. **The sampling path is a straight line in mel space.** MSE_k / [(1 - t_k)^2 MSE_0] = 0.985 +- 0.003 for k = 5..25
   (0.975 at k = 29). Per conversion, MSE_15 / MSE_0 = 0.246 (sd 0.001, range 0.244-0.252; exactly straight = 0.250)
   for every one of the 4,700 conversions. Each Euler step moves the state the same distance in the same direction;
   there is no wandering and no late correction. The 1.5 % shortfall means the first step is slightly longer than the
   others: the velocity at t = 0 overshoots a little and the rest of the path is straight.
2. **Step 0 is unit Gaussian noise, and a trace of it survives.** For noise independent of the result,
   MSE_0 = 1 + mean(x_30^2); measured MSE_0 - mean(x_30^2) = 0.30 (set 1) and 0.28 (set 2). The missing 0.7 is
   2 cov(z, x_30): the final mel is correlated with its own starting noise (correlation about 0.05), so a small
   noise residue survives the 30 steps. This is the same effect as the audible difference between the z1 and z2
   outputs of one source (max |dy| = 0.85 on the waveform in `run.log`).
3. **The starting noise does not change the curve.** The four noise seeds z1..z4 differ by at most 0.7 % (set 1)
   and 0.6 % (set 2) at any step.
4. **The spread between conversions is loudness, not path shape.** corr(MSE_0, mean x_30^2) = 1.000 in both sets.
   Set 1 (p10-p90 of MSE_0: 25-62) has a wider range of mel energies than set 2 (47-62), and set 2's curve lies
   about 17 % higher for the same reason. Dividing every curve by its own MSE_0 collapses both sets onto one line.
5. **Batching changed nothing.** Set 2 ran two noises per DiT pass (`batch 2`); the engine check in `run.log` gives
   batch 1 bit-identical, batch 2 within 0.07 % on the velocities and 1.3e-2 max on the waveform (float16 kernel
   effects). GPU peak 1.9 GB at batch 2, so the colleague's "if under 45 GB, double the batch" was not binding on GPU
   memory; if 45 GB meant disk, set 2 could be 10 conversations (`N_CONV=10`).

What the MSE curve cannot say: which steps carry speaker identity and which carry content. The step grids
(`../step_grids/`) suggest layout/content in steps 6-15 and harmonic/timbre structure in steps 16-25; the Aspect 1
run (`../../aspect1_mi/`) measures this with mutual-information estimates per step.
