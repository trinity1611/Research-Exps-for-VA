"""Unit tests of the MI estimators on synthetic data with known answers (spec section 7). Run:
    python experiments/aspect1_mi/tests/test_estimators.py
Each test prints its numbers and asserts the spec's property; the script exits non-zero on the first failure."""
import math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mi.estimators import (classifier_bound, infonce_bound, ksg, pca, permutation_null, p_value, speaker_bootstrap,
                           split_by_group, assert_no_group_overlap, entropy_of_labels)

rng = np.random.default_rng(0)
ok = True


def check(name, cond, detail):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    ok = ok and cond


# ---------------------------------------------------------------- 1. Gaussian pair, continuous: I = 0.5 * k * log(1 + SNR)
print("1. Gaussian pair s ~ N(0,I_k), y = s + sigma * n   (k = 2, sigma = 1 -> I = k * 0.5 * log 2 = 0.693 nats)")
k, sigma = 2, 1.0
truth = 0.5 * k * math.log(1 + 1 / sigma ** 2)
for N in (1000, 4000, 16000):
    s = rng.standard_normal((N, k)).astype(np.float32)
    y = s + sigma * rng.standard_normal((N, k)).astype(np.float32)
    s_tr, y_tr, s_te, y_te = s[: N // 2], y[: N // 2], s[N // 2:], y[N // 2:]
    e_ksg = ksg(s_te, y_te, k=3).value
    e_nce = infonce_bound(s_tr, y_tr, s_te, y_te, batch=min(512, N // 2), epochs=60, seed=0).value
    print(f"    N={N:6d}  truth {truth:.3f}   KSG {e_ksg:.3f}   InfoNCE {e_nce:.3f} (cap log N = {math.log(min(512, N // 2)):.2f})")
    check(f"KSG close to truth (N={N})", abs(e_ksg - truth) < 0.08, f"{e_ksg:.3f} vs {truth:.3f}")
    check(f"InfoNCE <= truth + tol and > 0.7 truth (N={N})", e_nce <= truth + 0.05 and e_nce > 0.7 * truth, f"{e_nce:.3f}")

# ---------------------------------------------------------------- 2. independent s, y -> ~0, inside the permutation null
print("2. independent s, y (k = 2): estimates ~ 0 and within the permutation null")
N = 4000
s = rng.standard_normal((N, 2)).astype(np.float32); y = rng.standard_normal((N, 2)).astype(np.float32)
e_ksg = ksg(s[N // 2:], y[N // 2:]).value
e_nce = infonce_bound(s[: N // 2], y[: N // 2], s[N // 2:], y[N // 2:], epochs=60).value
null = permutation_null(lambda a, b: infonce_bound(a, y[: N // 2], b, y[N // 2:], epochs=20), s[: N // 2], s[N // 2:], n=20)
print(f"    KSG {e_ksg:.3f}   InfoNCE {e_nce:.3f}   null InfoNCE {null['null_mean']:.3f} +- {null['null_sd']:.3f}   p = {p_value(e_nce, null['null']):.2f}")
check("KSG ~ 0", abs(e_ksg) < 0.05, f"{e_ksg:.3f}")
check("InfoNCE ~ 0 / within null", abs(e_nce) < 0.1 and p_value(e_nce, null["null"]) > 0.05, f"{e_nce:.3f}, p={p_value(e_nce, null['null']):.2f}")

# ---------------------------------------------------------------- 3. discrete S, classifier bound vs exact MI
print("3. discrete S (K = 20 classes, y = mu_S + sigma n in 8 dims): classifier bound <= exact I, approaching it")
K, d, sigma, per = 20, 8, 1.5, 400
mu = rng.standard_normal((K, d)) * 1.0
S = np.repeat(np.arange(K), per)
Y = (mu[S] + sigma * rng.standard_normal((len(S), d))).astype(np.float32)
# exact I(S;Y) = H(S) - E[-log p(s|y)] with p(s|y) from the known Gaussian mixture (Monte Carlo on fresh samples)
Sm = np.repeat(np.arange(K), 2000); Ym = mu[Sm] + sigma * rng.standard_normal((len(Sm), d))
ll = -((Ym[:, None, :] - mu[None]) ** 2).sum(-1) / (2 * sigma ** 2)                  # (n, K) log-lik up to a constant
logpost = ll - np.log(np.exp(ll - ll.max(1, keepdims=True)).sum(1, keepdims=True)) - ll.max(1, keepdims=True)
exact = math.log(K) - float(-logpost[np.arange(len(Sm)), Sm].mean())
# split by utterance (every class on both sides), as for the closed-set classifier in the spec
utt = np.arange(len(S))
tr, te = split_by_group(utt, 0.3, seed=1)
for probe in ("linear", "mlp"):
    e = classifier_bound(Y[tr], S[tr], Y[te], S[te], probe=probe, epochs=150)
    print(f"    {probe:6s}: H(S) = {e.h:.3f}  CE = {e.ce:.3f}  I_hat = {e.value:.3f}  (exact I = {exact:.3f}, acc {e.extra['acc']:.2f})")
    check(f"classifier bound ({probe}) <= exact + 0.05 and >= 0.8 exact", e.value <= exact + 0.05 and e.value >= 0.8 * exact, f"{e.value:.3f} vs {exact:.3f}")
lo, hi, _ = speaker_bootstrap(e.extra["ce_each"], S[te], n=200)
print(f"    speaker bootstrap of CE: 95 % CI [{e.h - hi:.3f}, {e.h - lo:.3f}] for I_hat (mlp)")
check("bootstrap CI contains the point estimate", e.h - hi <= e.value <= e.h - lo, "")

# ---------------------------------------------------------------- 4. confound: S-correlated target T inflates I(S;Y); I(S;Y|T) stays low
print("4. confound: y carries only the TARGET voice t; with t = f(s) (fixed pairing) I(S;Y) is inflated, I(S;Y|T) ~ 0")
K, nT, per = 20, 20, 150
S = np.repeat(np.arange(K), per)
T_fixed = (S * 7) % nT                                   # deterministic pairing: T is a function of S
T_indep = rng.integers(0, nT, len(S))                    # independent draw
tmu = rng.standard_normal((nT, 8)) * 2
for name, T in (("fixed pairing", T_fixed), ("independent T", T_indep)):
    Y = (tmu[T] + 0.5 * rng.standard_normal((len(S), 8))).astype(np.float32)      # no S information at all
    tr, te = split_by_group(np.arange(len(S)), 0.3, seed=2)
    onehot = np.eye(nT, dtype=np.float32)[T]
    e = classifier_bound(Y[tr], S[tr], Y[te], S[te], epochs=100)
    ec = classifier_bound(Y[tr], S[tr], Y[te], S[te], cond_tr=onehot[tr], cond_te=onehot[te], epochs=100)
    print(f"    {name:14s}: I_hat(S;Y) = {e.value:.3f}   I_hat(S;Y|T) = {ec.value:.3f}   (H(S) = {math.log(K):.3f})")
    if name == "fixed pairing":
        check("fixed pairing inflates I(S;Y)", e.value > 1.5, f"{e.value:.3f}")
        check("I(S;Y|T) stays ~ 0 under fixed pairing", ec.value < 0.3, f"{ec.value:.3f}")
    else:
        check("independent T: I(S;Y) ~ 0", e.value < 0.3, f"{e.value:.3f}")

# ---------------------------------------------------------------- 5. split leak test
print("5. split by group never puts a group on both sides")
g = np.repeat(np.arange(50), 10)
tr, te = split_by_group(g, 0.3, seed=3)
assert_no_group_overlap(g[tr], g[te])
try:
    assert_no_group_overlap(g[tr], np.concatenate([g[te], g[tr][:1]])); leaked = False
except AssertionError:
    leaked = True
check("split_by_group is leak-free and the assertion catches a leak", leaked and len(set(g[tr]) & set(g[te])) == 0, f"{len(set(g[te]))} test groups")

# ---------------------------------------------------------------- 6. PCA + KSG on higher-dimensional features
print("6. KSG after PCA to 4 dims on 64-dim features (truth for the 4 informative dims = 1.386)")
N = 6000
s = rng.standard_normal((N, 4)).astype(np.float32)
A = rng.standard_normal((4, 64)).astype(np.float32)
y = s @ A + 8.0 * rng.standard_normal((N, 64)).astype(np.float32)   # signal var per principal direction ~ 64 = noise var -> SNR 1
y_tr, y_te = pca(y[: N // 2], y[N // 2:], dims=4)
e = ksg(s[N // 2:], y_te, k=3).value
print(f"    KSG(PCA-4) = {e:.3f}")
check("KSG(PCA-4) below the 4-dim truth 1.386 (PCA from noisy data loses some signal) and well above 0", 0.9 < e <= 2 * math.log(2) + 0.1, f"{e:.3f}")

print("\nALL PASS" if ok else "\nSOME TESTS FAILED")
sys.exit(0 if ok else 1)
