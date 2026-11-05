# SPEC: Per-denoising-step Mutual Information for Seed-VC (speaker S, content C)

Audience: Claude Code. Implement the pipeline below. Items marked **[VERIFY]** depend on the exact Seed-VC variant/code; inspect the repo and adapt, do not guess.

---

## 0. Summary of the design

Goal: measure how much **source-speaker** information (S) and **content** information (C) is present in the Seed-VC output at each flow-matching denoising step t.

1. **Variables.** S = source speaker, C = content (sentence/phone), T = target (reference) speaker, X = source audio, Y_t = output representation at step t. Chain: S -> X -> Y_t, with T and noise x0 as independent side inputs.
2. **Target must be independent of S.** Sample T per conversion from a pool by a rule that ignores S (P(T|S)=P(T)), exclude T=S, log T. Otherwise a fixed S->T pairing makes a classifier "detect S" by recognising the target voice, inflating I(S;Y). Report both I(S;Y) and I(S;Y|T).
3. **Discrete -> continuous S.** Replace label S by an embedding s = phi(x) in R^k from an independent ASV encoder. Use continuous MI estimators (InfoNCE/KSG). Optionally estimate intrinsic dimension and local tangent spaces.
4. **C discrete** = finite label set: utterance-level (sentence ID, H(C)=log M) or frame-level (phone label from forced alignment, ~40-50 classes).
5. **Estimators are lower bounds**: classifier (Barber-Agakov) bound for discrete labels, InfoNCE for continuous/open-set. Always evaluate on held-out data, split by speaker/utterance, never by frame.
6. **DPI ceiling**: I(S;Y) <= I(S;X). Estimate I(S;X) with the same estimator on the original audio; use it to normalise leakage and as a sanity check (Y estimate must not exceed X estimate).
7. **Conditioning is injected at every step**, so Y_t must be defined explicitly (section 3).

---

## 1. Data design

- K speakers (>= 50-100), all reading the same M sentences (>= 20-50) -> crossed S x C grid so S and C are independent under the sampling distribution. Optionally add natural speech.
- Target pool: speakers disjoint from source speakers. For each source utterance, draw T uniformly at random (independent of S), with R noise seeds (R >= 3).
- Metadata table (parquet/CSV), one row per generated output:
  `utt_id, src_spk, sentence_id, target_spk, target_ref_path, seed, split, steps_saved`
- Splits: speaker-disjoint for open-set/InfoNCE evaluators; utterance-disjoint (same speakers) for closed-set classifiers. Fix splits and seeds before looking at results.

## 2. Generation (Seed-VC, unmodified)

- Run Seed-VC inference with ODE step count K_steps (e.g. 25-50). **[VERIFY]** sampler, CFG scale, conditioning path (content features, reference mel prompt, style embedding).
- Convention: t=0 pure noise, t=1 data. Save, for each step t_k:
  - `x_t` (latent/mel state)
  - `v = v_theta(x_t, t, u)`
  - `xhat1_t = x_t + (1 - t) * v`  (one-step clean estimate)
- Save conditioning `u` once per (utt, target).
- Storage: optionally save only a subset grid of steps (e.g. 8 values), fp16, one file per output.

## 3. Defining Y_t (must be explicit; log which one is used)

- **Y_t^A = x_t** alone (raw state). Expect I(S;Y_t) ~ 0 at t=0.
- **Y_t^B = xhat1_t -> vocoder -> waveform -> independent encoders.** Primary choice; raw noisy x_t is out-of-distribution for ASV/ASR.
- **Y_t^C = (x_t, u)**: includes conditioning. The valid ceiling for the final output is I(S; x_t, u), not I(S; x_t). I(S;Y_t) need not be monotone in t.

Run B as the primary analysis; run A as the t=0 sanity check.

## 4. Feature extraction (independent of Seed-VC)

- Speaker features f_S(y): >= 2 independent ASV families (e.g. ECAPA-TDNN, a WavLM-based SV model). **[VERIFY]** availability. Do not rely on Seed-VC's own style encoder as the only evaluator; use it only as a second view.
- Content features f_C(y): ASR encoder / phonetic posteriorgram (e.g. Whisper encoder, wav2vec2 CTC); frame-level phone labels via forced alignment of the source text.
- Continuous-S embedding: s = f_S(x_source), computed on the original audio.

## 5. Estimators (same estimator + same capacity at every t)

Notation: K speakers, H(S)=log K for uniform S, M sentences, N batch size.

### 5.1 Classifier (Barber-Agakov) bound, discrete S or C
```
I_hat(S; Y_t) = H(S) - CE_heldout(t)        # lower bound
CE_heldout = mean over held-out samples of -log q(s | f_S(Y_t))
```
- q = linear softmax probe AND small MLP (report both).
- Content: `I_hat(C;Y_t) = H(C) - CE_heldout` (utterance-level H(C)=log M; frame-level H(C)=entropy of the phone prior from training labels).
- Conditional on target: `I_hat(S; Y_t | T) = H(S|T) - CE_heldout(q(s | f_S(Y_t), onehot(T)))`. With T independent of S, H(S|T)=H(S).
- Conditional on content: `I_hat(S; Y_t | C)` using sentence ID as an extra classifier input.

### 5.2 InfoNCE bound, continuous / open-set S
```python
# s: (N,k) source speaker embeddings, y: (N,d) features of Y_t; (s_i,y_i) paired
scores = critic(s[:, None, :], y[None, :, :])        # (N,N)
loss   = F.cross_entropy(scores, torch.arange(N))
mi_lb  = math.log(N) - loss.item()                    # evaluate on HELD-OUT batches only
```
- Critic: separable (two MLP encoders + dot product) or bilinear. Fixed architecture, optimiser, epochs, N for all t.
- Capped at log N: use N >= 512; report the cap.
- Cross-check in low dimension (PCA to <= 10 dims) with KSG (k = 3-5) on the same pairs.
- Optionally MINE (Donsker-Varadhan): `E_joint[f] - log E_marg[exp f]`, marginal samples by shuffling y across pairs.

### 5.3 Ceilings and normalisation
```
ceiling_S = I_hat(S; X)        # same estimator on ORIGINAL audio features
ceiling_C = I_hat(C; X)
leak_frac_S(t) = I_hat(S;Y_t) / ceiling_S
util_frac_C(t) = I_hat(C;Y_t) / ceiling_C
```
Assert `I_hat(S;Y_t) <= ceiling_S` (within CI tolerance); otherwise flag estimator inconsistency.

### 5.4 Sanity checks and nulls
- t=0, Y^A: I_hat ~ 0.
- Permutation null: shuffle labels and re-estimate (>= 100 shuffles) per t; report p-value.
- Positive control: Y = X (should recover the ceiling). Negative control: Y = pure noise.
- Confound demonstration: a deliberately S-correlated target pairing should inflate I(S;Y) while I(S;Y|T) stays low; then use the independent design.

### 5.5 Uncertainty
- Speaker-level bootstrap (resample speakers, not utterances/frames), >= 200 resamples; report 95% CIs.
- Never split by frame; never let one speaker appear in both train and test for open-set evaluation.

## 6. Continuous-S geometry (optional diagnostics)

- Intrinsic dimension of {s}: TwoNN and MLE estimators.
- Local tangent spaces: PCA on k-NN of each s.
- Global distances: Isomap / kNN-graph geodesics.
- Chart: autoencoder/VAE on s.
- Interpolation test between two real speakers' embeddings; check output validity (ASR WER, a naturalness predictor such as UTMOS) and flag invalid regions.

## 7. Unit tests (required before real data)

- Synthetic Gaussian pair (s, y = s + noise): known `I = 0.5*log(1+SNR)` per dimension; check that classifier/InfoNCE/KSG estimates are <= truth and approach it as N grows.
- Independent s, y: estimate ~ 0 and within the permutation null.
- Confound test: construct S-correlated T; verify I(S;Y) inflates while I(S;Y|T) stays low.
- Split-leak test: assert no speaker overlap between train and test for open-set runs.

## 8. Outputs

- `results.parquet`: columns `t, estimator, encoder, probe, target(S|C), cond(None|T|C), I_hat, ci_lo, ci_hi, null_mean, null_p, ceiling, frac`.
- Plots: I_hat(S;Y_t) and I_hat(C;Y_t) vs t with CIs, ceiling lines, null band; leakage-vs-utility scatter across t.
- Report which Y_t definition, which encoders, probe capacity, and N were used, and state that all values are **lower bounds**.

## 9. Caveats to record

- Lower bounds: a low value means little information OR a weak probe. Use multiple encoders and probes.
- From N samples you cannot certify MI above ~log N.
- MI estimates describe what the chosen evaluators can extract; ASV EER/identification against a stated threat model remains the ground truth for privacy.
- The T-independence design is part of the threat model (a pairing-aware attacker is a different threat model).
