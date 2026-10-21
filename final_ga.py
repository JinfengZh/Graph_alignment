# -*- coding: utf-8 -*-

# ============================================================
# GRAPH ALIGNMENT
#
# G1 = {a,b,c,d}      (n nodes, unified space N = n+k)
# G2 = {a,b,c,y}      (y replaces d : same features, same links,
#                      new dimension)
#
# Figure 1 : Accuracy vs p_replace  (with vs without β, λ fixe)
# Figure 2 : Accuracy vs λ          (+ adaptive overlay)


import os, re
import time
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected
from joblib import Parallel, delayed

PATH = "./results/0510/"
os.makedirs(PATH, exist_ok=True)

N_JOBS = 32

CONFIG = dict(
    dataset = "Cora",
    seed    = 42,
    k       = 2,
    alpha   = 0.5,
    topk    = 16,
    n_seeds = 5,
)

DATASETS = [
    "Texas","Actor","Cora","Chameleon","CiteSeer", 
    "Amazon-Computers", "Amazon-Photo",  "Squirrel",
      "PubMed",
    "Cornell", "Wisconsin",
    "WikiCS",
]

P_FIXE_VALUES = [0.3,0.5]
NOISE_S_VALUES = np.round(np.linspace(0.0, 0.5, 11), 2)
NOISE_F_VALUES = np.round(np.linspace(0.0, 0.5, 11), 2)


def load_tg_dataset(name):
    tr = NormalizeFeatures()

    if name in ("Cora", "CiteSeer", "PubMed"):
        from torch_geometric.datasets import Planetoid
        ds = Planetoid("data/Planetoid", name, transform=tr)

    elif name in ("Squirrel", "Chameleon"):
        from torch_geometric.datasets import WikipediaNetwork
        ds = WikipediaNetwork("data/WikipediaNetwork", name, transform=tr)

    elif name == "Actor":
        from torch_geometric.datasets import Actor
        ds = Actor("data/Actor", transform=tr)

    elif name in ("Amazon-Computers", "Amazon-Photo"):
        from torch_geometric.datasets import Amazon
        amazon_name = name.replace("Amazon-", "")
        ds = Amazon("data/Amazon", amazon_name, transform=tr)

    elif name in ("Computers", "Photo"):
        from torch_geometric.datasets import Amazon
        ds = Amazon("data/Amazon", name, transform=tr)

    elif name in ("CS", "Physics"):
        from torch_geometric.datasets import Coauthor
        ds = Coauthor("data/Coauthor", name, transform=tr)

    elif name in ("Cornell", "Texas", "Wisconsin"):
        from torch_geometric.datasets import WebKB
        ds = WebKB("data/WebKB", name, transform=tr)

    elif name == "WikiCS":
        from torch_geometric.datasets import WikiCS
        ds = WikiCS("data/WikiCS", transform=tr)

    else:
        raise ValueError(f"Unknown dataset: {name}")

    d = ds[0]
    d.edge_index = to_undirected(d.edge_index, num_nodes=d.num_nodes).cpu().numpy()
    d.x = d.x.cpu().numpy()
    return d

# ============================================================
# ADJ + FEATURES
# ============================================================

def build_adj(src, dst, n):
    A = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n)).tocsr()
    A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()
    return A

def contextual_features(X):
    X  = np.asarray(X, dtype=np.float64)
    Rv = X.sum(axis=1)
    ni = X.sum(axis=0)
    N = X.sum()
    eps = 1e-12

    pv = X / np.maximum(Rv[:, None], eps)
    p_notv = (ni[None, :] - X) / np.maximum((N - Rv)[:, None], eps)

    valid = (pv > 0) & (pv < 1) & (p_notv > 0) & (p_notv < 1)

    lo = (
        np.log(np.clip(pv, eps, 1 - eps) / (1 - np.clip(pv, eps, 1 - eps)))
        - np.log(np.clip(p_notv, eps, 1 - eps) / (1 - np.clip(p_notv, eps, 1 - eps)))
    )

    gamma = lo * np.abs(pv - p_notv)
    C = np.where(valid & (gamma > 0), gamma, 0.0)

    s = C.sum(axis=1, keepdims=True)
    s[s == 0] = 1

    return (C / s).astype(np.float32)

# ============================================================
# MAKE VIEWS — unified space N = n + k
# ============================================================

def make_views(data, p_replace, edge_drop_prob=0.0, feat_noise_prob=0.0, seed=42):
    rng = np.random.default_rng(seed)
    n = data.num_nodes
    ei = np.asarray(data.edge_index)

    k = int(round(p_replace * n))

    replaced = (
        np.sort(rng.choice(n, size=k, replace=False))
        if k > 0 else np.array([], dtype=np.int64)
    )

    common = np.array(
        sorted(set(range(n)) - set(replaced.tolist())),
        dtype=np.int64
    )

    N = n + k

    mapping = np.arange(N, dtype=np.int64)
    for j, old in enumerate(replaced):
        mapping[old] = n + j

    A1_orig = build_adj(ei[0], ei[1], n)

    A1 = (
        sp.block_diag([A1_orig, sp.csr_matrix((k, k))]).tocsr()
        if k > 0 else A1_orig.copy()
    )

    s2, d2 = mapping[ei[0]], mapping[ei[1]]

    if edge_drop_prob > 0:
        keep = rng.random(len(s2)) > edge_drop_prob
        s2, d2 = s2[keep], d2[keep]

    A2 = build_adj(s2, d2, N)

    X_ctx = contextual_features(data.x)
    F = X_ctx.shape[1]

    X_A = np.zeros((N, F), dtype=np.float32)
    X_A[:n] = X_ctx

    X_B = np.zeros((N, F), dtype=np.float32)
    X_B[common] = X_ctx[common]

    for j, old in enumerate(replaced):
        X_B[n + j] = X_ctx[old]

    if feat_noise_prob > 0:
        all_G2 = (
            np.concatenate([common, np.arange(k, dtype=np.int64) + n])
            if k > 0 else common
        )

        mask = rng.random(len(all_G2)) < feat_noise_prob
        noisy = all_G2[mask]

        if len(noisy):
            X_B[noisy] = X_ctx[rng.integers(0, n, size=len(noisy))]

    queries = np.concatenate([common, replaced]) if k > 0 else common

    targets = (
        np.concatenate([common, np.arange(k, dtype=np.int64) + n])
        if k > 0 else common
    )

    return A1, A2, X_A, X_B, queries, targets, N

# ============================================================
# β
# ============================================================

_BETA_CACHE = {}
def compute_beta(A):
    A_csr = A.tocsr()
    key = (A_csr.indices.tobytes(), A_csr.indptr.tobytes(), A_csr.shape)
    if key in _BETA_CACHE:
        return _BETA_CACHE[key]
    n = A_csr.shape[0]
    adj = [set(A_csr.indices[A_csr.indptr[i]:A_csr.indptr[i+1]].tolist())
           for i in range(n)]
    rows_e, cols_e = A_csr.nonzero()
    common_set, common_size = {}, {}
    for z, v in zip(rows_e, cols_e):
        if (z, v) in common_set: continue
        s = adj[z] & adj[v]
        common_set[(z, v)] = s;          common_set[(v, z)] = s
        common_size[(z, v)] = len(s);    common_size[(v, z)] = len(s)
    rows, cols, vals = [], [], []
    for z, v in zip(rows_e, cols_e):
        sset = common_set[(z, v)]
        sz = len(sset)
        if sz == 0: continue
        acc = 0.0
        for u in sset:
            su = common_size.get((z, u), 0)
            denom = sz + su - 1
            if denom > 0:
                acc += (su - 1) / denom
        rows.append(z); cols.append(v); vals.append(acc / sz)
    result = sp.coo_matrix((vals, (rows, cols)), shape=A.shape).tocsr()
    _BETA_CACHE[key] = result
    return result

# ============================================================
# BOUNDED PROPAGATION
# ============================================================

_PROP_CACHE = {}
def clear_runtime_cache():
    _BETA_CACHE.clear()
    _PROP_CACHE.clear()

def propagate(A, k=2, alpha=0.5, use_beta=True):
    A_csr = A.tocsr()
    key = (A_csr.indices.tobytes(), A_csr.indptr.tobytes(),
           A_csr.shape, k, alpha, use_beta)
    if key in _PROP_CACHE:
        return _PROP_CACHE[key]
    n = A.shape[0]
    I = sp.eye(n, format="csr")
    M = I + alpha * (A.multiply(compute_beta(A)) if use_beta else A)
    W = I.copy()
    for _ in range(k):
        W = M @ W
        rs = np.asarray(W.sum(axis=1)).ravel(); rs[rs == 0] = 1
        W = sp.diags(1 / rs) @ W
    W.eliminate_zeros()
    _PROP_CACHE[key] = W
    return W

def row_topk(W, topk):
    W = W.tocsr()
    n = W.shape[0]

    rs, cs, vs = [], [], []

    for i in range(n):
        s, e = W.indptr[i], W.indptr[i + 1]
        idx, dat = W.indices[s:e], W.data[s:e]

        if not len(dat):
            continue

        keep = (
            np.argpartition(dat, -topk)[-topk:]
            if len(dat) > topk else np.arange(len(dat))
        )

        rs.extend([i] * len(keep))
        cs.extend(idx[keep])
        vs.extend(dat[keep])

    Wk = sp.coo_matrix((vs, (rs, cs)), shape=W.shape).tocsr()
    nrm = np.asarray(Wk.sum(axis=1)).ravel()

    return (sp.diags(1 / np.maximum(nrm, 1e-12)) @ Wk).tocsr()



def _sparse_l2(W):
    W = W.tocsr().copy()

    for i in range(W.shape[0]):
        s, e = W.indptr[i], W.indptr[i + 1]
        d = W.data[s:e]
        nrm = np.sqrt((d ** 2).sum())

        if nrm > 1e-12:
            W.data[s:e] = d / nrm

    return W

def _row_l2(X):
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(nrm, 1e-12)

def base_similarities(W_A, W_B, X_A, X_B, queries):
    S_s = (
        _sparse_l2(W_A)[queries, :] @ _sparse_l2(W_B).T
    ).toarray().astype(np.float32)

    S_f = (
        _row_l2(X_A)[queries, :] @ _row_l2(X_B).T
    ).astype(np.float32)

    return S_s, S_f

def fuse_fixed(S_s, S_f, lam):
    return float(lam) * S_s + (1.0 - float(lam)) * S_f

def fuse_adaptive(S_s, S_f, topk=None, tau=10.0):
    # confiance = 1 - entropie normalisée de softmax
    def confidence(S):
        Z = tau * S - (tau * S).max(axis=1, keepdims=True)
        P = np.exp(Z); P /= P.sum(axis=1, keepdims=True)
        H = -np.sum(P * np.log(P + 1e-12), axis=1)
        return 1.0 - H / np.log(S.shape[1])
    cs, cc = confidence(S_s), confidence(S_f)
    lam = cs / np.maximum(cs + cc, 1e-12)
    S = lam[:, None] * S_s + (1.0 - lam[:, None]) * S_f
    return S, lam



def evaluate(S, targets):
    pred = np.argmax(S, axis=1)
    acc = float((pred == targets).mean())

    srt = np.argsort(-S, axis=1)
    ranks = (srt == targets[:, None]).argmax(axis=1) + 1

    return {
        "acc": float(acc),
        "mrr": float((1.0 / ranks).mean()),
        "hits@5": float((ranks <= 5).mean()),
        "hits@10": float((ranks <= 10).mean()),
    }

def run_one(
    data,
    p_replace,
    lam,
    edge_drop_prob,
    feat_noise_prob,
    cfg,
    seed,
    use_beta,
    fusion="fixed"
):
    A1, A2, X_A, X_B, queries, targets, N = make_views(
        data,
        p_replace,
        edge_drop_prob,
        feat_noise_prob,
        seed=seed
    )

    if not len(queries):
        return {"acc": 0, "mrr": 0, "hits@5": 0, "hits@10": 0}, 0.0

    W_A = row_topk(
        propagate(A1, cfg["k"], cfg["alpha"], use_beta),
        cfg["topk"]
    )

    W_B = row_topk(
        propagate(A2, cfg["k"], cfg["alpha"], use_beta),
        cfg["topk"]
    )

    S_s, S_f = base_similarities(W_A, W_B, X_A, X_B, queries)

    if fusion == "adaptive":
        S, alpha = fuse_adaptive(S_s, S_f, cfg["topk"])
        return evaluate(S, targets), float(alpha.mean())

    return evaluate(fuse_fixed(S_s, S_f, lam), targets), float(lam)

# ============================================================
# FIGURE 1  (with vs without β, λ=1, no noise)
# ============================================================

def experiment1(data, cfg, lam=1.0):
    print(f"\n=== Figure 1 : Accuracy vs p_replace (λ={lam}, no noise) ===")

    p_values = np.arange(0.0, 1.1, 0.1).round(2)
    out = {"with_beta": {}, "without_beta": {}}

    for p in p_values:
        for key, ub in [("with_beta", True), ("without_beta", False)]:
            accs = [
                run_one(data, p, lam, 0.0, 0.0, cfg, s, ub)[0]["acc"]
                for s in range(cfg["n_seeds"])
            ]

            out[key][p] = (float(np.mean(accs)), float(np.std(accs)))

        print(
            f"  p={p:.1f}  "
            f"with_β={out['with_beta'][p][0]:.3f}  "
            f"without_β={out['without_beta'][p][0]:.3f}"
        )

    return out, p_values

def plot_figure1(res, p_values, cfg, save_path, lam=0.5):
    _style()

    fig, ax = plt.subplots(figsize=(5.5, 3.6))

    for key, label, color, mk, ls in [
        ("with_beta", r"With $\beta$ (structural dependence)", "#2166ac", "o", "-"),
        ("without_beta", r"Without $\beta$ (uniform propagation)", "#d6604d", "s", "--"),
    ]:
        m = np.array([res[key][p][0] for p in p_values])
        e = np.array([res[key][p][1] for p in p_values])

        ax.plot(
            p_values,
            m,
            f"{mk}{ls}",
            color=color,
            label=label,
            lw=1.6,
            ms=4.5
        )

        ax.fill_between(
            p_values,
            m - e,
            m + e,
            alpha=0.12,
            color=color
        )

    ax.set_xlabel("Fraction of replaced nodes $p$", fontsize=9)
    ax.set_ylabel("Accuracy (Hits@1)", fontsize=9)

    ax.set_title(
        f"{cfg['dataset']} — Alignment accuracy vs. replacement ratio  "
        f"($\\lambda={lam}$, no noise)",
        fontsize=8.5
    )

    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    print(f"  Saved: {save_path}")

    plt.close()

def _style():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.4,
    })

# ============================================================
# MAIN
# ============================================================

def safe(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

def run_dataset(ds):
    cfg = CONFIG.copy()
    cfg["dataset"] = ds

    tag = safe(ds)

    dataset_path = os.path.join(PATH, tag)
    os.makedirs(dataset_path, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"Dataset : {ds}")
    print("=" * 80)

    data = load_tg_dataset(ds)

    print(
        f"Dataset : {ds} | "
        f"Nodes : {data.num_nodes} | "
        f"Edges : {data.edge_index.shape[1] // 2}"
    )

    # ------------------------------------------------------------
    # Figure 1
    # ------------------------------------------------------------

    res1, p_vals = experiment1(data, cfg, lam=1.0)

    plot_figure1(
        res1,
        p_vals,
        cfg,
        os.path.join(dataset_path, f"{tag}_fig1_preplaced.pdf"),
        lam=1.0
    )

    # ------------------------------------------------------------
    # Figure 2: all combinations of P_FIXE, NOISE_S, NOISE_F
    # ------------------------------------------------------------

    all_rows = []
    adaptive_metrics_rows = []
    best_fixed_metrics_rows = []
    lambda_metrics_rows = []

    for P_FIXE in P_FIXE_VALUES:
        for NOISE_S in NOISE_S_VALUES:
            for NOISE_F in NOISE_F_VALUES:

                print(
                    f"\n--- Figure 2 ({ds}) | "
                    f"S:{NOISE_S}, F:{NOISE_F}, p={P_FIXE} ---"
                )

                lambdas = np.round(np.linspace(0.0, 1.0, 21), 2)

                acc_m, acc_e = [], []
                fixed_metrics_by_lambda = []

                for l in lambdas:
                    lambda_metrics = []

                    for s in range(cfg["n_seeds"]):
                        metric = run_one(
                            data,
                            P_FIXE,
                            l,
                            NOISE_S,
                            NOISE_F,
                            cfg,
                            s,
                            True
                        )[0]

                        lambda_metrics.append(metric)

                    fixed_metrics_by_lambda.append({
                        "lambda": float(l),
                        "metrics": lambda_metrics,
                    })

                    # Save aggregated metrics for every fixed lambda
                    lambda_metrics_rows.append({
                        "dataset": ds,
                        "p_replace": P_FIXE,
                        "noise_s": NOISE_S,
                        "noise_f": NOISE_F,
                        "lambda": float(l),
                        "n_seeds": cfg["n_seeds"],
                        "acc_mean": float(np.mean([x["acc"] for x in lambda_metrics])),
                        "acc_std": float(np.std([x["acc"] for x in lambda_metrics])),
                        "mrr_mean": float(np.mean([x["mrr"] for x in lambda_metrics])),
                        "mrr_std": float(np.std([x["mrr"] for x in lambda_metrics])),
                        "hits5_mean": float(np.mean([x["hits@5"] for x in lambda_metrics])),
                        "hits5_std": float(np.std([x["hits@5"] for x in lambda_metrics])),
                        "hits10_mean": float(np.mean([x["hits@10"] for x in lambda_metrics])),
                        "hits10_std": float(np.std([x["hits@10"] for x in lambda_metrics])),
                    })

                    acc_m.append(np.mean([x["acc"] for x in lambda_metrics]))
                    acc_e.append(np.std([x["acc"] for x in lambda_metrics]))

                m = np.array(acc_m)
                e = np.array(acc_e)

                ad_scores, ad_lams = [], []
                ad_metrics = []
                ad_times = []

                for s in range(cfg["n_seeds"]):
                    t0 = time.perf_counter()
                    r, lm = run_one(
                        data,
                        P_FIXE,
                        0.0,
                        NOISE_S,
                        NOISE_F,
                        cfg,
                        s,
                        True,
                        fusion="adaptive"
                    )
                    t1 = time.perf_counter()

                    ad_scores.append(r["acc"])
                    ad_lams.append(lm)
                    ad_metrics.append(r)
                    ad_times.append(t1 - t0)

                ad_mean = float(np.mean(ad_scores))
                ad_std = float(np.std(ad_scores))
                lam_mean = float(np.mean(ad_lams))

                best_idx = int(np.argmax(m))
                best_lam = float(lambdas[best_idx])

                # ------------------------------------------------------------
                # Cold-cache runtime measurement for the selected best lambda
                # ------------------------------------------------------------
                best_fixed_times = []
                for s in range(cfg["n_seeds"]):
                    clear_runtime_cache()
                    t0 = time.perf_counter()
                    run_one(
                        data,
                        P_FIXE,
                        best_lam,
                        NOISE_S,
                        NOISE_F,
                        cfg,
                        s,
                        True
                    )[0]
                    t1 = time.perf_counter()
                    best_fixed_times.append(t1 - t0)

                best_fixed_time_mean = float(np.mean(best_fixed_times))
                best_fixed_time_std = float(np.std(best_fixed_times))

                print(
                    f"  >>> Best fixed λ = {best_lam:.2f}  "
                    f"acc = {m[best_idx]:.4f} ± {e[best_idx]:.4f}"
                )

                print(
                    f"  >>> Adaptive λ    "
                    f"acc = {ad_mean:.4f} ± {ad_std:.4f}  "
                    f"(mean λ = {lam_mean:.3f})"
                )

                idx_lam0 = int(np.argmin(np.abs(np.asarray(lambdas, dtype=float) - 0.0)))
                idx_lam1 = int(np.argmin(np.abs(np.asarray(lambdas, dtype=float) - 1.0)))
                lam0_acc = float(m[idx_lam0])
                lam1_acc = float(m[idx_lam1])

                print(f"  >>> λ=0 (sem)     : {lam0_acc:.4f}")
                print(f"  >>> λ=1 (struct)  : {lam1_acc:.4f}")

                all_rows.append({
                    "dataset": ds,
                    "p_replace": P_FIXE,
                    "noise_s": NOISE_S,
                    "noise_f": NOISE_F,
                    "best_fixed_lambda": best_lam,
                    "best_fixed_acc_mean": float(m[best_idx]),
                    "best_fixed_acc_std": float(e[best_idx]),
                    "adaptive_acc_mean": ad_mean,
                    "adaptive_acc_std": ad_std,
                    "adaptive_lambda_mean": lam_mean,
                    "lambda_0_semantic_acc": lam0_acc,
                    "lambda_1_structural_acc": lam1_acc,
                })

                adaptive_metrics_rows.append({
                    "p_replace": P_FIXE,
                    "noise_s": NOISE_S,
                    "noise_f": NOISE_F,
                    "acc_mean": float(np.mean([x["acc"] for x in ad_metrics])),
                    "acc_std": float(np.std([x["acc"] for x in ad_metrics])),
                    "mrr_mean": float(np.mean([x["mrr"] for x in ad_metrics])),
                    "mrr_std": float(np.std([x["mrr"] for x in ad_metrics])),
                    "hits5_mean": float(np.mean([x["hits@5"] for x in ad_metrics])),
                    "hits5_std": float(np.std([x["hits@5"] for x in ad_metrics])),
                    "hits10_mean": float(np.mean([x["hits@10"] for x in ad_metrics])),
                    "hits10_std": float(np.std([x["hits@10"] for x in ad_metrics])),
                    "time": float(np.mean(ad_times)),
                    "time_std": float(np.std(ad_times)),
                })

                best_fixed_metrics = fixed_metrics_by_lambda[best_idx]["metrics"]

                best_fixed_metrics_rows.append({
                    "p_replace": P_FIXE,
                    "noise_s": NOISE_S,
                    "noise_f": NOISE_F,
                    "best_fixed_lambda": best_lam,
                    "acc_mean": float(np.mean([x["acc"] for x in best_fixed_metrics])),
                    "acc_std": float(np.std([x["acc"] for x in best_fixed_metrics])),
                    "mrr_mean": float(np.mean([x["mrr"] for x in best_fixed_metrics])),
                    "mrr_std": float(np.std([x["mrr"] for x in best_fixed_metrics])),
                    "hits5_mean": float(np.mean([x["hits@5"] for x in best_fixed_metrics])),
                    "hits5_std": float(np.std([x["hits@5"] for x in best_fixed_metrics])),
                    "hits10_mean": float(np.mean([x["hits@10"] for x in best_fixed_metrics])),
                    "hits10_std": float(np.std([x["hits@10"] for x in best_fixed_metrics])),
                    "time": best_fixed_time_mean,
                    "time_std": best_fixed_time_std,
                })

                _style()

                plt.figure(figsize=(5.5, 3.6))

                plt.plot(
                    lambdas,
                    m,
                    'o-',
                    color='#e67e22',
                    lw=1.5,
                    ms=4,
                    label=f"Fixed fusion  (noise S={NOISE_S}, F={NOISE_F})"
                )

                plt.fill_between(
                    lambdas,
                    m - e,
                    m + e,
                    alpha=0.15,
                    color='#e67e22'
                )

                plt.scatter(
                    [best_lam],
                    [m[best_idx]],
                    s=110,
                    facecolors='none',
                    edgecolors='#2166ac',
                    linewidths=2.0,
                    zorder=5,
                    label=fr"Best $\lambda^\star$={best_lam:.2f}  (acc={m[best_idx]:.3f})"
                )

                plt.axvline(
                    best_lam,
                    color='#2166ac',
                    ls=':',
                    lw=1.0,
                    alpha=0.6
                )

                plt.axhline(
                    ad_mean,
                    color='#27ae60',
                    ls='--',
                    lw=1.3,
                    label=fr"Adaptive λ (acc={ad_mean:.3f}, $\bar\lambda$={lam_mean:.2f})"
                )

                plt.fill_between(
                    [0, 1],
                    [ad_mean - ad_std] * 2,
                    [ad_mean + ad_std] * 2,
                    alpha=0.10,
                    color='#27ae60'
                )

                plt.xlabel(r"Fusion weight $\lambda$  (0 = semantic, 1 = structure)")
                plt.ylabel("Accuracy (Hits@1)")

                plt.title(
                    f"Figure 2 : Fusion sweep  "
                    f"(p={P_FIXE}, {ds}, S={NOISE_S}, F={NOISE_F})"
                )

                plt.ylim(-0.02, 1.05)
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                fig_name = (
                    f"{tag}_fig2_"
                    f"p{P_FIXE:.2f}_"
                    f"S{NOISE_S:.2f}_"
                    f"F{NOISE_F:.2f}.pdf"
                )

                plt.savefig(
                    os.path.join(dataset_path, fig_name),
                    dpi=300
                )

                plt.close()

    # ------------------------------------------------------------
    # Save summary CSV for this dataset
    # ------------------------------------------------------------

    try:
        import pandas as pd

        df = pd.DataFrame(all_rows)

        csv_path = os.path.join(
            dataset_path,
            f"{tag}_summary_results.csv"
        )

        df.to_csv(csv_path, index=False)

        adaptive_metrics_df = pd.DataFrame(adaptive_metrics_rows)
        adaptive_metrics_csv_path = os.path.join(
            dataset_path,
            f"{tag}_adaptive_metrics_summary.csv"
        )
        adaptive_metrics_df.to_csv(adaptive_metrics_csv_path, index=False)

        best_fixed_metrics_df = pd.DataFrame(best_fixed_metrics_rows)
        best_fixed_metrics_csv_path = os.path.join(
            dataset_path,
            f"{tag}_best_fixed_lambda_metrics_summary.csv"
        )
        best_fixed_metrics_df.to_csv(best_fixed_metrics_csv_path, index=False)

        lambda_metrics_df = pd.DataFrame(lambda_metrics_rows)
        lambda_metrics_csv_path = os.path.join(
            dataset_path,
            f"{tag}_all_lambda_metrics_summary.csv"
        )
        lambda_metrics_df.to_csv(lambda_metrics_csv_path, index=False)

        print(f"\nSaved summary CSV: {csv_path}")
        print(f"Saved adaptive metrics summary CSV: {adaptive_metrics_csv_path}")
        print(f"Saved best fixed lambda metrics summary CSV: {best_fixed_metrics_csv_path}")
        print(f"Saved all-lambda metrics summary CSV: {lambda_metrics_csv_path}")

    except ImportError:
        print("\nPandas is not installed. Summary CSV was not saved.")

    return ds

if __name__ == "__main__":
    print(f"Running datasets in parallel with N_JOBS = {N_JOBS}")

    Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(run_dataset)(ds)
        for ds in DATASETS
    )
