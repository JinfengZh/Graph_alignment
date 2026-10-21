# -*- coding: utf-8 -*-
"""
REGAL baseline for the RQ3 protocol .

This implementation follows the REGAL/xNetMF paper and the authors' reference code:
  - K-hop structural identity from logarithmically binned degree distributions
  - landmark-based Nyström approximation / xNetMF embeddings
  - nearest-neighbor alignment in the shared embedding space

It is adapted to the SAME noisy graph views used by final_ga.py / gnn1.py:
  - node replacement
  - edge deletion
  - semantic-feature corruption
  - unified candidate space N = n + round(p*n)

For the high-dimensional real-valued semantic signatures used in the current code,
we use cosine attribute distance. REGAL explicitly allows Euclidean or cosine
attribute distance for real-valued attributes (Sec. 3.1 of the REGAL paper).

Default RQ3 protocol (matching the latest manuscript):
  1) edge-noise sweep:    p=0.3, eta_f=0.3, eta_s in {0, .05, ..., .50}
  2) feature-noise sweep: p=0.3, eta_s=0.3, eta_f in {0, .05, ..., .50}

By default REGAL is evaluated on the SAME held-out ~80% query set used by the
GNN baselines. REGAL itself is unsupervised; the 20% split is used only to make
the evaluation population identical to the GNN evaluation population.
"""

import os
import re
import time
import math
import numpy as np
import scipy.sparse as sp



# ============================================================
# CONFIG
# ============================================================

PATH = "./regal_rq3/"
os.makedirs(PATH, exist_ok=True)

CONFIG = dict(
    n_seeds=5,
    train_anchor_ratio=0.2,

    # REGAL / xNetMF settings from Heimann et al. (CIKM 2018)
    regal_max_layer=2,          # K
    regal_delta=0.01,           # discount factor delta (called alpha in ref code)
    regal_gamma_struct=1.0,
    regal_gamma_attr=1.0,
    regal_bucket_base=2,
    regal_landmark_mult=10.0,   # p = floor(10 * log2(n_active))
    regal_landmark_seed=42,


    regal_attribute_distance="cosine",

    # For exact comparability with GNN evaluation, keep "heldout".
    # Set to "all" only if you intentionally want to reproduce the current
    # proposed-method evaluation on all source queries.
    eval_scope="heldout",

    # Chunk size for alignment evaluation (controls memory, not results).
    eval_chunk_size=256,
)

DATASETS = ["Texas","Chameleon", "CiteSeer", "Cora", "Actor"]

P_REPLACE_RQ3 = 0.30
EDGE_NOISE_VALUES = np.round(np.linspace(0.0, 0.5, 11), 2)
FEATURE_NOISE_VALUES = np.round(np.linspace(0.0, 0.5, 11), 2)
FIXED_EDGE_NOISE = 0.30
FIXED_FEATURE_NOISE = 0.30


# ============================================================
# DATA — same loaders as current code
# ============================================================

def load_tg_dataset(name):
    from torch_geometric.transforms import NormalizeFeatures
    from torch_geometric.utils import to_undirected
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


def build_adj(src, dst, n):
    A = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n)).tocsr()
    A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def contextual_features(X):
    """Exactly the semantic preprocessing used in the current proposed/GNN code."""
    X = np.asarray(X, dtype=np.float64)
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


def make_views(data, p_replace, edge_drop_prob=0.0, feat_noise_prob=0.0, seed=42):
    """Same view construction as final_ga.py / gnn1.py."""
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
        dtype=np.int64,
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
    Fdim = X_ctx.shape[1]

    X_A = np.zeros((N, Fdim), dtype=np.float32)
    X_A[:n] = X_ctx

    X_B = np.zeros((N, Fdim), dtype=np.float32)
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




def split_train_eval_pairs(queries, targets, train_anchor_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(queries))
    rng.shuffle(idx)

    n_train = max(1, int(round(train_anchor_ratio * len(idx))))
    train_idx = idx[:n_train]
    eval_idx = idx[n_train:]

    if len(eval_idx) == 0:
        eval_idx = train_idx

    return train_idx, eval_idx




def _degree_bucket(degree, base, n_buckets):
    if degree <= 0:
        return None
    b = int(math.log(degree, base))
    return min(max(b, 0), n_buckets - 1)


def regal_structural_features(A, max_layer=2, delta=0.01, bucket_base=2):
    """
    xNetMF structural identity using exact-hop degree distributions.

    This mirrors the authors' reference implementation, including the 0-hop
    self layer and weighting each layer by delta**layer.
    """
    A = A.tocsr()
    n = A.shape[0]
    degrees = np.diff(A.indptr).astype(np.int64)
    max_degree = int(degrees.max()) if n else 0

    if max_degree <= 1:
        n_buckets = 1
    else:
        n_buckets = int(math.log(max_degree, bucket_base)) + 1

    F = np.zeros((n, n_buckets), dtype=np.float32)

    indptr = A.indptr
    indices = A.indices

    for u in range(n):
        # layer 0: node itself (as in the official reference implementation)
        b = _degree_bucket(int(degrees[u]), bucket_base, n_buckets)
        if b is not None:
            F[u, b] += 1.0

        visited = {u}
        frontier = set(indices[indptr[u]:indptr[u + 1]].tolist())
        frontier.discard(u)
        visited.update(frontier)

        for layer in range(1, max_layer + 1):
            if not frontier:
                break

            weight = float(delta) ** layer
            for v in frontier:
                bv = _degree_bucket(int(degrees[v]), bucket_base, n_buckets)
                if bv is not None:
                    F[u, bv] += weight

            if layer == max_layer:
                break

            next_frontier = set()
            for v in frontier:
                next_frontier.update(indices[indptr[v]:indptr[v + 1]].tolist())
            next_frontier.difference_update(visited)
            visited.update(next_frontier)
            frontier = next_frontier

    return F


# ============================================================
# REGAL STEPS 2-3 — xNetMF + ALIGNMENT
# ============================================================

def _row_l2(X):
    X = np.asarray(X, dtype=np.float32)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(nrm, 1e-12)


def _euclidean_to_landmarks(F, landmark_rows):
    """Pairwise Euclidean distance from every row of F to landmark rows."""
    F = np.asarray(F, dtype=np.float32)
    L = F[landmark_rows]
    f2 = np.sum(F * F, axis=1, keepdims=True)
    l2 = np.sum(L * L, axis=1, keepdims=True).T
    d2 = f2 + l2 - 2.0 * (F @ L.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2, dtype=np.float32)


def _cosine_attr_distance_to_landmarks(X, landmark_rows):
    """
    1 - cosine similarity for real-valued node attributes.
    REGAL explicitly permits cosine distance for real-valued attributes.
    Zero vectors have cosine similarity 0 to all landmarks, hence distance 1.
    """
    Xn = _row_l2(X)
    sim = Xn @ Xn[landmark_rows].T
    np.clip(sim, -1.0, 1.0, out=sim)
    return 1.0 - sim


def choose_regal_landmarks(active_rows, n_landmarks, seed=42):
    """
    Sample landmarks uniformly from REAL nodes only.

    The unified N=n+k representation contains padded isolated positions that
    are not actual graph entities. They remain alignment candidates (to keep the
    same candidate space as the current experiments), but are not allowed to
    waste REGAL landmark slots.
    """
    active_rows = np.asarray(active_rows, dtype=np.int64)
    if len(active_rows) == 0:
        raise ValueError("No active nodes available for REGAL landmarks.")
    n_landmarks = min(int(n_landmarks), len(active_rows))
    rng = np.random.RandomState(seed)
    return rng.permutation(active_rows)[:n_landmarks]


def xnetmf_from_similarity(C, landmark_rows):
    """
    Official xNetMF Nyström embedding:
      W^† = pinv(C[landmarks])
      W^† = U Sigma V^T
      Y = C U Sigma^(1/2)
    followed by row L2 normalization.
    """
    W = C[landmark_rows, :]
    W_pinv = np.linalg.pinv(W)
    U, s, _ = np.linalg.svd(W_pinv, full_matrices=False)
    Wfac = U * np.sqrt(np.maximum(s, 0.0))[None, :]
    Y = C @ Wfac
    return _row_l2(Y).astype(np.float32)


def regal_embeddings(A1, A2, X_A, X_B, queries, targets, cfg):
    """Build joint xNetMF embeddings for two graph views."""
    N = A1.shape[0]
    if A2.shape[0] != N:
        raise ValueError("This RQ3 adapter expects both views in the same unified N-space.")

    # REGAL jointly embeds a block-diagonal combination of the two graphs.
    A = sp.block_diag((A1, A2), format="csr")

    F_struct = regal_structural_features(
        A,
        max_layer=cfg["regal_max_layer"],
        delta=cfg["regal_delta"],
        bucket_base=cfg["regal_bucket_base"],
    )

    # Real source nodes are exactly `queries` (all n original entities);
    # real target nodes are exactly `targets` (common IDs + replacement IDs).
    active_rows = np.concatenate([
        np.asarray(queries, dtype=np.int64),
        N + np.asarray(targets, dtype=np.int64),
    ])

    # Paper/reference default: p = floor(10 log2(n)), where n is the number
    # of actual nodes across the two graphs. Use active count, not padded slots.
    n_active = len(active_rows)
    p = max(1, int(math.floor(cfg["regal_landmark_mult"] * math.log(max(n_active, 2), 2))))
    landmark_rows = choose_regal_landmarks(
        active_rows,
        p,
        seed=cfg["regal_landmark_seed"],
    )

    struct_dist = _euclidean_to_landmarks(F_struct, landmark_rows)
    total_dist = cfg["regal_gamma_struct"] * struct_dist

    attr_mode = cfg.get("regal_attribute_distance", "cosine")
    if attr_mode == "cosine":
        X = np.vstack([X_A, X_B]).astype(np.float32, copy=False)
        attr_dist = _cosine_attr_distance_to_landmarks(X, landmark_rows)
        total_dist = total_dist + cfg["regal_gamma_attr"] * attr_dist
    elif attr_mode in (None, "none"):
        pass
    else:
        raise ValueError(
            f"Unsupported regal_attribute_distance={attr_mode!r}. "
            "Use 'cosine' or 'none'."
        )

    # Same exp(-distance) kernel as REGAL.
    C = np.exp(-total_dist).astype(np.float32)
    Y = xnetmf_from_similarity(C, landmark_rows)

    return Y[:N], Y[N:], int(len(landmark_rows))


def evaluate_embedding_alignment(E1, E2, queries, targets, chunk_size=256):
    """
    Evaluate normalized REGAL embeddings without materializing the full
    |Q| x N similarity matrix. Ranking uses cosine similarity, which is
    monotonic with Euclidean distance for L2-normalized embeddings.
    """
    E1 = _row_l2(E1)
    E2 = _row_l2(E2)
    queries = np.asarray(queries, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)

    n = len(queries)
    if n == 0:
        return {"acc": 0.0, "mrr": 0.0, "hits@5": 0.0, "hits@10": 0.0}

    correct1 = 0
    reciprocal_rank_sum = 0.0
    hits5 = 0
    hits10 = 0
    candidate_ids = np.arange(E2.shape[0], dtype=np.int64)[None, :]

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        q = queries[start:end]
        t = targets[start:end]

        S = E1[q] @ E2.T
        pred = np.argmax(S, axis=1)
        correct1 += int(np.sum(pred == t))

        target_scores = S[np.arange(len(q)), t]
        # Deterministic rank: higher scores first; for exact ties, lower
        # candidate index first (consistent with np.argmax's first-index rule).
        greater = np.sum(S > target_scores[:, None], axis=1)
        tied_before = np.sum(
            (S == target_scores[:, None]) & (candidate_ids < t[:, None]),
            axis=1,
        )
        ranks = 1 + greater + tied_before

        reciprocal_rank_sum += float(np.sum(1.0 / ranks))
        hits5 += int(np.sum(ranks <= 5))
        hits10 += int(np.sum(ranks <= 10))

    return {
        "acc": float(correct1 / n),
        "mrr": float(reciprocal_rank_sum / n),
        "hits@5": float(hits5 / n),
        "hits@10": float(hits10 / n),
    }


def run_one_regal(data, p_replace, edge_drop_prob, feat_noise_prob, cfg, seed):
    A1, A2, X_A, X_B, queries, targets, _ = make_views(
        data,
        p_replace,
        edge_drop_prob,
        feat_noise_prob,
        seed=seed,
    )

    if len(queries) == 0:
        return {
            "acc": 0.0, "mrr": 0.0, "hits@5": 0.0, "hits@10": 0.0,
            "n_landmarks": 0, "n_eval": 0,
        }

    if cfg.get("eval_scope", "heldout") == "heldout":
        _, eval_idx = split_train_eval_pairs(
            queries,
            targets,
            train_anchor_ratio=cfg["train_anchor_ratio"],
            seed=seed,
        )
        eval_queries = queries[eval_idx]
        eval_targets = targets[eval_idx]
    elif cfg.get("eval_scope") == "all":
        eval_queries = queries
        eval_targets = targets
    else:
        raise ValueError("eval_scope must be 'heldout' or 'all'")

    E1, E2, n_landmarks = regal_embeddings(
        A1, A2, X_A, X_B, queries, targets, cfg
    )

    metrics = evaluate_embedding_alignment(
        E1,
        E2,
        eval_queries,
        eval_targets,
        chunk_size=cfg["eval_chunk_size"],
    )
    metrics["n_landmarks"] = int(n_landmarks)
    metrics["n_eval"] = int(len(eval_queries))
    return metrics


# ============================================================
# RQ3
# ============================================================

def safe(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def aggregate_condition(dataset, sweep, p_replace, noise_s, noise_f, metrics, times, cfg):
    return {
        "dataset": dataset,
        "model": "REGAL",
        "sweep": sweep,
        "p_replace": float(p_replace),
        "noise_s": float(noise_s),
        "noise_f": float(noise_f),
        "eval_scope": cfg["eval_scope"],
        "attr_distance": cfg["regal_attribute_distance"],
        "regal_K": int(cfg["regal_max_layer"]),
        "regal_delta": float(cfg["regal_delta"]),
        "regal_gamma_struct": float(cfg["regal_gamma_struct"]),
        "regal_gamma_attr": float(cfg["regal_gamma_attr"]),
        "acc_mean": float(np.mean([x["acc"] for x in metrics])),
        "acc_std": float(np.std([x["acc"] for x in metrics])),
        "mrr_mean": float(np.mean([x["mrr"] for x in metrics])),
        "mrr_std": float(np.std([x["mrr"] for x in metrics])),
        "hits5_mean": float(np.mean([x["hits@5"] for x in metrics])),
        "hits5_std": float(np.std([x["hits@5"] for x in metrics])),
        "hits10_mean": float(np.mean([x["hits@10"] for x in metrics])),
        "hits10_std": float(np.std([x["hits@10"] for x in metrics])),
        "time": float(np.mean(times)),
        "time_std": float(np.std(times)),
        "n_landmarks_mean": float(np.mean([x["n_landmarks"] for x in metrics])),
        "n_eval_mean": float(np.mean([x["n_eval"] for x in metrics])),
    }


def run_dataset(dataset):
    cfg = CONFIG.copy()
    tag = safe(dataset)
    out_dir = os.path.join(PATH, tag)
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 88)
    print(f"REGAL RQ3 | Dataset: {dataset}")
    print("=" * 88)

    data = load_tg_dataset(dataset)
    print(
        f"Nodes={data.num_nodes} | edges={data.edge_index.shape[1] // 2} | "
        f"features={data.x.shape[1]}"
    )

    rows = []
    seed_rows = []

    # Cache the center condition (eta_s=eta_f=0.3), which belongs to both sweeps.
    cache = {}

    def run_condition(sweep, noise_s, noise_f):
        key_base = (float(P_REPLACE_RQ3), float(noise_s), float(noise_f))
        metrics_by_seed = []
        times_by_seed = []

        for seed in range(cfg["n_seeds"]):
            key = key_base + (seed,)
            if key in cache:
                r, elapsed = cache[key]
            else:
                t0 = time.perf_counter()
                r = run_one_regal(
                    data=data,
                    p_replace=P_REPLACE_RQ3,
                    edge_drop_prob=noise_s,
                    feat_noise_prob=noise_f,
                    cfg=cfg,
                    seed=seed,
                )
                elapsed = time.perf_counter() - t0
                cache[key] = (r, elapsed)

            metrics_by_seed.append(r)
            times_by_seed.append(elapsed)

            seed_rows.append({
                "dataset": dataset,
                "model": "REGAL",
                "sweep": sweep,
                "seed": seed,
                "p_replace": P_REPLACE_RQ3,
                "noise_s": float(noise_s),
                "noise_f": float(noise_f),
                "eval_scope": cfg["eval_scope"],
                "acc": r["acc"],
                "mrr": r["mrr"],
                "hits@5": r["hits@5"],
                "hits@10": r["hits@10"],
                "time": elapsed,
                "n_landmarks": r["n_landmarks"],
                "n_eval": r["n_eval"],
            })

        row = aggregate_condition(
            dataset,
            sweep,
            P_REPLACE_RQ3,
            noise_s,
            noise_f,
            metrics_by_seed,
            times_by_seed,
            cfg,
        )
        rows.append(row)

        print(
            f"[{sweep:12s}] p={P_REPLACE_RQ3:.2f} "
            f"S={noise_s:.2f} F={noise_f:.2f} | "
            f"Hits@1={row['acc_mean']:.4f}±{row['acc_std']:.4f} | "
            f"MRR={row['mrr_mean']:.4f} | "
            f"H@5={row['hits5_mean']:.4f} | H@10={row['hits10_mean']:.4f} | "
            f"time={row['time']:.3f}s"
        )

    # RQ3(a): edge-noise sweep, feature noise fixed at 0.3
    for noise_s in EDGE_NOISE_VALUES:
        run_condition("edge_noise", float(noise_s), FIXED_FEATURE_NOISE)

    # RQ3(b): feature-noise sweep, edge noise fixed at 0.3
    for noise_f in FEATURE_NOISE_VALUES:
        run_condition("feature_noise", FIXED_EDGE_NOISE, float(noise_f))

    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        csv_path = os.path.join(out_dir, f"{tag}_regal_rq3_results.csv")
        df.to_csv(csv_path, index=False)

        df_seed = pd.DataFrame(seed_rows)
        seed_csv_path = os.path.join(out_dir, f"{tag}_regal_rq3_seed_results.csv")
        df_seed.to_csv(seed_csv_path, index=False)

        # GNN-compatible columns for easy concatenation with
        # *_gnn_all_models_summary_results.csv.
        gnn_cols = [
            "dataset", "model", "p_replace", "noise_s", "noise_f",
            "acc_mean", "acc_std", "mrr_mean", "mrr_std",
            "hits5_mean", "hits5_std", "hits10_mean", "hits10_std",
            "time", "time_std",
        ]
        df_gnn_compatible = df[gnn_cols].copy()
        compat_path = os.path.join(out_dir, f"{tag}_regal_gnn_compatible_results.csv")
        df_gnn_compatible.to_csv(compat_path, index=False)

        # One row per RQ3 sweep, useful for Table 4-style summaries.
        df_sweep = (
            df.groupby(["dataset", "model", "sweep"], as_index=False)
            .agg({
                "acc_mean": "mean",
                "mrr_mean": "mean",
                "hits5_mean": "mean",
                "hits10_mean": "mean",
                "time": "mean",
            })
        )
        sweep_path = os.path.join(out_dir, f"{tag}_regal_rq3_sweep_means.csv")
        df_sweep.to_csv(sweep_path, index=False)

        print(f"Saved: {csv_path}")
        print(f"Saved: {seed_csv_path}")
        print(f"Saved: {compat_path}")
        print(f"Saved: {sweep_path}")

    except ImportError:
        print("Pandas not installed; CSV files were not saved.")


if __name__ == "__main__":
    for ds in DATASETS:
        run_dataset(ds)
