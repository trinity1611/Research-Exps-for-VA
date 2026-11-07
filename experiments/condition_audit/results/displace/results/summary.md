## D4 identity in the condition (closed set, 120 speakers)
                hbar linear dim  1024: I_hat  3.89 [3.67, 4.10]  acc  78.3 %  (9 s)
                hbar mlp    dim  1024: I_hat  3.80 [3.60, 3.99]  acc  74.8 %  (1 s)
                hstd linear dim  1024: I_hat  3.05 [2.73, 3.37]  acc  59.6 %  (0 s)
                hstd mlp    dim  1024: I_hat  2.86 [2.57, 3.15]  acc  55.1 %  (1 s)
              hdelta linear dim  1024: I_hat  3.30 [3.02, 3.59]  acc  63.8 %  (0 s)
              hdelta mlp    dim  1024: I_hat  3.15 [2.89, 3.38]  acc  61.0 %  (1 s)
    hbar+hstd+hdelta linear dim  3072: I_hat  3.72 [3.46, 3.99]  acc  73.7 %  (0 s)
    hbar+hstd+hdelta mlp    dim  3072: I_hat  3.61 [3.35, 3.87]  acc  71.4 %  (1 s)
             mu_mean linear dim   384: I_hat  3.84 [3.65, 4.03]  acc  74.6 %  (0 s)
             mu_mean mlp    dim   384: I_hat  3.76 [3.57, 3.94]  acc  71.6 %  (1 s)
              mu_std linear dim   384: I_hat  3.82 [3.61, 4.08]  acc  77.0 %  (0 s)
              mu_std mlp    dim   384: I_hat  3.66 [3.43, 3.92]  acc  71.2 %  (1 s)
      mu_mean+mu_std linear dim   768: I_hat  4.01 [3.80, 4.23]  acc  79.0 %  (0 s)
      mu_mean+mu_std mlp    dim   768: I_hat  3.86 [3.68, 4.05]  acc  75.3 %  (1 s)
## D4 identity coordinate theta = U_S^T (hbar - mu): LDA on 84 training speakers, probes on the 36 held-out speakers
  full hbar on the held-out speakers (H = 3.57): I_hat 3.29 acc 90.7 %
  lda k=  1: coordinate I_hat  0.03 (acc  3.7 %)   residual I_hat  3.29 (acc 90.7 %)
  lda k=  2: coordinate I_hat  0.23 (acc 10.5 %)   residual I_hat  3.27 (acc 90.1 %)
  lda k=  4: coordinate I_hat  0.72 (acc 18.5 %)   residual I_hat  3.24 (acc 88.9 %)
  lda k=  8: coordinate I_hat  1.76 (acc 37.0 %)   residual I_hat  3.25 (acc 88.3 %)
  lda k= 16: coordinate I_hat  2.72 (acc 74.7 %)   residual I_hat  3.20 (acc 88.9 %)
  lda k= 32: coordinate I_hat  3.26 (acc 92.6 %)   residual I_hat  3.24 (acc 89.5 %)
  lda k= 64: coordinate I_hat  3.14 (acc 88.3 %)   residual I_hat  3.26 (acc 89.5 %)
  lda k= 83: coordinate I_hat  3.21 (acc 86.4 %)   residual I_hat  3.31 (acc 92.0 %)
  pca k=  1: coordinate I_hat -0.04 (acc  4.3 %)   residual I_hat  3.30 (acc 90.1 %)
  pca k=  2: coordinate I_hat  0.18 (acc  9.9 %)   residual I_hat  3.32 (acc 92.6 %)
  pca k=  4: coordinate I_hat  0.42 (acc 14.8 %)   residual I_hat  3.33 (acc 92.6 %)
  pca k=  8: coordinate I_hat  1.23 (acc 27.8 %)   residual I_hat  3.30 (acc 92.6 %)
  pca k= 16: coordinate I_hat  2.21 (acc 54.9 %)   residual I_hat  3.30 (acc 92.6 %)
  pca k= 32: coordinate I_hat  3.10 (acc 84.6 %)   residual I_hat  3.34 (acc 91.4 %)
  pca k= 64: coordinate I_hat  3.33 (acc 92.0 %)   residual I_hat  3.21 (acc 90.1 %)
  pca k= 83: coordinate I_hat  3.32 (acc 93.2 %)   residual I_hat  3.17 (acc 90.1 %)
## D6 identity from duration and rate only (closed set)
         dur_s linear: I_hat -0.02 [-0.02, -0.02]  acc   0.2 %
         dur_s mlp   : I_hat -0.02 [-0.02, -0.02]  acc   0.9 %
    pause_frac linear: I_hat  0.02 [-0.03, 0.07]  acc   0.9 %
    pause_frac mlp   : I_hat  0.17 [0.10, 0.23]  acc   1.1 %
   chars_per_s linear: I_hat -0.02 [-0.02, -0.02]  acc   0.6 %
   chars_per_s mlp   : I_hat -0.02 [-0.02, -0.02]  acc   0.9 %
          all4 linear: I_hat  0.02 [-0.03, 0.07]  acc   1.7 %
          all4 mlp   : I_hat  0.03 [-0.02, 0.07]  acc   0.9 %
  log chars/s: mean 2.574, var 0.3140 (Var rho for the rate branch); pause frac mean 0.073
## D10 context-only floor: speaker from the ASR transcript (TF-IDF -> SVD-256 -> probe, closed set)
   char1-3 linear (vocab 4369): I_hat  0.02 [0.00, 0.05]  acc   2.4 %
   char1-3 mlp    (vocab 4369): I_hat -0.02 [-0.03, -0.01]  acc   0.6 %
     word1 linear (vocab 409): I_hat -0.01 [-0.01, -0.01]  acc   1.5 %
     word1 mlp    (vocab 409): I_hat -0.02 [-0.02, -0.01]  acc   0.9 %
## D15 intrinsic dimension and curvature of the XLS-R utterance means
  n: 1796
  dim: 1024
  pca_dims_90: 209
  pca_dims_95: 346
  twonn_id: 27.289
  mle_id_k10_20: 21.120
  twonn_id_speaker_means: 12.235
  within_between_ratio_R: 1.401
  local_dim90_median: 35.000
  local_dim90_p10: 34.000
  local_dim90_p90: 36.000
  tangent_rotation_neighbours_deg_mean: 53.361
  tangent_rotation_neighbours_deg_p90: 61.384
  tangent_vs_global_lda_deg_mean: 72.603
  tangent_vs_global_pca_deg_mean: 55.647
  random_subspace_vs_pca_deg: 85.312
  d_loc: 10
  k_neighbours: 50
  n_anchors: 150
## D8 cosine ASV on the step-30 outputs (closed-set split of Aspect 1; enrol = speaker centroid of the training utterances)
  wavlm    orig->orig (reference)                : EER   8.5 %  top-1  57.7 %  (539 trials, 120 speakers)
  wavlm    ignorant: enrol orig, trial anon      : EER  48.2 %  top-1   1.6 %  (1617 trials, 120 speakers)
  wavlm    lazy-informed: enrol anon, trial anon : EER  48.5 %  top-1   1.5 %  (1617 trials, 120 speakers)
  campplus orig->orig (reference)                : EER   1.8 %  top-1  91.3 %  (539 trials, 120 speakers)
  campplus ignorant: enrol orig, trial anon      : EER  42.7 %  top-1   2.9 %  (1617 trials, 120 speakers)
  campplus lazy-informed: enrol anon, trial anon : EER  41.1 %  top-1   3.8 %  (1617 trials, 120 speakers)
