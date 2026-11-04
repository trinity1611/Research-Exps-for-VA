"""Mutual-information lower bounds used by the per-step analysis (spec section 5).

All estimates are LOWER BOUNDS of I(.;.) in nats and are always evaluated on HELD-OUT data:

  classifier_bound   Barber-Agakov: I(S;Y) >= H(S) - CE_heldout, S discrete, linear softmax or small MLP probe
  infonce_bound      I(S;Y) >= log N - CE_heldout of a separable critic over N-sample batches (capped at log N)
  ksg                Kraskov-Stoegbauer-Grassberger k-NN estimator (not a bound; cross-check in <= 10 dims)
  permutation_null   the same estimator on shuffled labels / pairings
  speaker_bootstrap  95 % CI by resampling SPEAKERS of the held-out set

Splits are passed in by the caller (by speaker or by utterance, never by frame; see splits.py).
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np


# ----------------------------------------------------------------------------------------------- helpers
def entropy_of_labels(labels):
    """H of the empirical label prior, in nats."""
    _, c = np.unique(labels, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def _onehot(idx, n):
    out = np.zeros((len(idx), n), dtype=np.float32)
    out[np.arange(len(idx)), idx] = 1
    return out


def _standardize(train, *others):
    mu, sd = train.mean(0, keepdims=True), train.std(0, keepdims=True) + 1e-6
    return [(x - mu) / sd for x in (train,) + others]


@dataclass
class Estimate:
    value: float            # the bound / estimate in nats
    ce: float = float("nan")
    h: float = float("nan")  # entropy of the label prior (classifier) or log N (InfoNCE)
    extra: dict = None


# ----------------------------------------------------------------------------------------------- classifier bound
def classifier_bound(y_tr, s_tr, y_te, s_te, probe="linear", cond_tr=None, cond_te=None, seed=0, epochs=200,
                     hidden=256, weight_decay=1e-4, device=None, return_model=False):
    """I(S;Y) >= H(S) - CE_heldout.  s_*: integer labels 0..K-1.  cond_*: optional extra inputs (one-hot T or C) that
    turn the bound into I(S;Y|cond) (with H(S|cond) estimated from the training labels).
    probe = 'linear' (softmax regression) or 'mlp' (one hidden layer). Both are trained with the same optimiser,
    epochs and weight decay at every t (same capacity everywhere, spec 5)."""
    import torch
    import torch.nn as nn
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    K = int(max(s_tr.max(), s_te.max())) + 1
    y_tr, y_te = _standardize(np.asarray(y_tr, np.float32), np.asarray(y_te, np.float32))
    if cond_tr is not None:
        y_tr = np.concatenate([y_tr, cond_tr.astype(np.float32)], 1)
        y_te = np.concatenate([y_te, cond_te.astype(np.float32)], 1)
        h = conditional_entropy(s_tr, cond_tr.argmax(1))
    else:
        h = entropy_of_labels(s_tr)
    torch.manual_seed(seed)
    d = y_tr.shape[1]
    net = nn.Linear(d, K) if probe == "linear" else nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, K))
    net = net.to(dev)
    X, S = torch.from_numpy(y_tr).to(dev), torch.from_numpy(np.asarray(s_tr, np.int64)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=weight_decay)
    bs = min(256, len(X))
    for ep in range(epochs):
        perm = torch.randperm(len(X), device=dev)
        net.train()
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            loss = nn.functional.cross_entropy(net(X[idx]), S[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        logits = net(torch.from_numpy(y_te).to(dev))
        ce_each = nn.functional.cross_entropy(logits, torch.from_numpy(np.asarray(s_te, np.int64)).to(dev), reduction="none").cpu().numpy()
        acc = float((logits.argmax(1).cpu().numpy() == s_te).mean())
    ce = float(ce_each.mean())
    est = Estimate(value=h - ce, ce=ce, h=h, extra={"acc": acc, "ce_each": ce_each, "probe": probe, "K": K})
    return (est, net) if return_model else est


def conditional_entropy(s, c):
    """H(S|C) from joint counts, in nats."""
    s, c = np.asarray(s), np.asarray(c)
    h = 0.0
    n = len(s)
    for cv in np.unique(c):
        m = c == cv
        h += m.sum() / n * entropy_of_labels(s[m])
    return float(h)


# ----------------------------------------------------------------------------------------------- InfoNCE bound
def infonce_bound(s_tr, y_tr, s_te, y_te, batch=512, epochs=50, hidden=256, out=64, seed=0, device=None, n_eval=50, lr=1e-3,
                  critic="separable"):
    """I(S;Y) >= log N - CE on held-out batches of N paired samples (s_i, y_i); s continuous (e.g. ASV embedding of the
    source audio), y continuous (features of Y_t). Critic: separable (two MLPs + dot product) or bilinear. The bound is
    capped at log N; the cap is returned in .h. Same architecture / optimiser / epochs / N at every t."""
    import torch
    import torch.nn as nn
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    s_tr, s_te = _standardize(np.asarray(s_tr, np.float32), np.asarray(s_te, np.float32))
    y_tr, y_te = _standardize(np.asarray(y_tr, np.float32), np.asarray(y_te, np.float32))
    N = min(batch, len(s_te), len(s_tr))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if critic == "separable":
        f = nn.Sequential(nn.Linear(s_tr.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, out)).to(dev)
        g = nn.Sequential(nn.Linear(y_tr.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, out)).to(dev)
        scale = nn.Parameter(torch.tensor(1.0, device=dev))
        params = list(f.parameters()) + list(g.parameters()) + [scale]
        score = lambda a, b: (f(a) @ g(b).T) * scale
    else:
        W = nn.Linear(y_tr.shape[1], s_tr.shape[1], bias=False).to(dev)
        params = list(W.parameters())
        score = lambda a, b: a @ W(b).T
    opt = torch.optim.Adam(params, lr=lr)
    S, Y = torch.from_numpy(s_tr).to(dev), torch.from_numpy(y_tr).to(dev)
    labels = torch.arange(N, device=dev)
    for ep in range(epochs):
        perm = torch.randperm(len(S), device=dev)
        for i in range(0, len(S) - N + 1, N):
            idx = perm[i:i + N]
            loss = nn.functional.cross_entropy(score(S[idx], Y[idx]), labels)
            opt.zero_grad(); loss.backward(); opt.step()
    # held-out: average CE over n_eval random batches of N held-out pairs
    St, Yt = torch.from_numpy(s_te).to(dev), torch.from_numpy(y_te).to(dev)
    ces = []
    with torch.no_grad():
        for _ in range(n_eval):
            idx = torch.from_numpy(rng.choice(len(St), N, replace=False)).to(dev)
            ces.append(float(nn.functional.cross_entropy(score(St[idx], Yt[idx]), labels)))
    ce = float(np.mean(ces))
    return Estimate(value=math.log(N) - ce, ce=ce, h=math.log(N), extra={"N": N, "critic": critic, "ce_sd": float(np.std(ces))})


# ----------------------------------------------------------------------------------------------- KSG
def ksg(x, y, k=3):
    """Kraskov et al. (2004) estimator 1 of I(X;Y) for continuous x (N,kx), y (N,ky); max-norm; nats.
    Use in low dimension (PCA to <= 10 dims first)."""
    from scipy.special import digamma
    from scipy.spatial import cKDTree
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    x = (x - x.mean(0)) / (x.std(0) + 1e-12)                                 # unit scale per dimension: the max-norm
    y = (y - y.mean(0)) / (y.std(0) + 1e-12)                                 # in the joint space must not be dominated by one side
    x = x + 1e-10 * np.random.default_rng(0).standard_normal(x.shape)       # break ties
    y = y + 1e-10 * np.random.default_rng(1).standard_normal(y.shape)
    N = len(x)
    xy = np.concatenate([x, y], 1)
    eps = cKDTree(xy).query(xy, k=k + 1, p=np.inf)[0][:, -1]
    tx, ty = cKDTree(x), cKDTree(y)
    nx = np.array([len(tx.query_ball_point(x[i], eps[i] - 1e-12, p=np.inf)) - 1 for i in range(N)])
    ny = np.array([len(ty.query_ball_point(y[i], eps[i] - 1e-12, p=np.inf)) - 1 for i in range(N)])
    v = digamma(k) + digamma(N) - np.mean(digamma(nx + 1) + digamma(ny + 1))
    return Estimate(value=float(v), extra={"k": k, "N": N})


def pca(train, *others, dims=10, seed=0):
    """Project to the top `dims` principal directions of the training features."""
    mu = train.mean(0, keepdims=True)
    u, s, vt = np.linalg.svd(train - mu, full_matrices=False)
    P = vt[:dims].T
    return [(x - mu) @ P for x in (train,) + others]


# ----------------------------------------------------------------------------------------------- nulls, bootstrap
def permutation_null(fn, labels_tr, labels_te, n=100, seed=0):
    """Re-run an estimator with the labels / pairings shuffled (train and test independently). `fn(l_tr, l_te)` must
    return an Estimate. Returns mean, sd, the null values and p = P(null >= observed) when `observed` is given later."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        vals.append(fn(rng.permutation(labels_tr), rng.permutation(labels_te)).value)
    vals = np.asarray(vals)
    return {"null_mean": float(vals.mean()), "null_sd": float(vals.std()), "null": vals}


def p_value(observed, null):
    null = np.asarray(null)
    return float((np.sum(null >= observed) + 1) / (len(null) + 1))


def speaker_bootstrap(per_sample_values, speakers, n=200, seed=0, reduce=np.mean):
    """95 % CI of a held-out statistic by resampling SPEAKERS (not samples). per_sample_values: e.g. -log q(s|y) per
    held-out sample; speakers: the speaker of each sample. Returns (lo, hi, values)."""
    rng = np.random.default_rng(seed)
    spk = np.asarray(speakers)
    uniq = np.unique(spk)
    groups = {u: np.flatnonzero(spk == u) for u in uniq}
    vals = []
    for _ in range(n):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([groups[u] for u in pick])
        vals.append(reduce(np.asarray(per_sample_values)[idx]))
    vals = np.asarray(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), vals


# ----------------------------------------------------------------------------------------------- splits
def split_by_group(groups, test_frac=0.3, seed=0):
    """Train/test indices with whole groups (speakers or utterances) on one side only; never splits a group."""
    rng = np.random.default_rng(seed)
    g = np.asarray(groups)
    uniq = rng.permutation(np.unique(g))
    n_te = max(1, int(round(len(uniq) * test_frac)))
    te_groups = set(uniq[:n_te].tolist())
    te = np.array([x in te_groups for x in g])
    assert not (set(g[te]) & set(g[~te])), "split leak: a group on both sides"
    return np.flatnonzero(~te), np.flatnonzero(te)


def assert_no_group_overlap(groups_tr, groups_te):
    leak = set(np.asarray(groups_tr).tolist()) & set(np.asarray(groups_te).tolist())
    assert not leak, f"split leak: {sorted(leak)[:5]} ... appear in both train and test"
