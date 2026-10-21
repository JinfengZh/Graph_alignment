# -*- coding: utf-8 -*-

# ============================================================
# GRAPH ALIGNMENT + PYTORCH GEOMETRIC GNN BASELINES (NO BETA IN GNN)
#
# Original setting:
# G1 = {a,b,c,d}      (n nodes, unified space N = n+k)
# G2 = {a,b,c,y}      (y replaces d : same features, same links,
#                      new dimension)
#
# Added setting:
# Run the same datasets / p_replace / noise_s / noise_f combinations
# with multiple PyG GNN encoders:
# GCNConv, SAGEConv, GATConv, GraphConv, GINConv, TAGConv,
# ChebConv, SGConv, ARMAConv.
# ============================================================

import os, re
import time
import numpy as np
import scipy.sparse as sp
import matplotlib, matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected

PATH = "./gnn1/new/"
os.makedirs(PATH, exist_ok=True)

CONFIG = dict(
    dataset = "Cora",
    seed    = 42,
    k       = 2,
    alpha   = 0.5,
    topk    = 16,
    n_seeds = 5,
    # GNN baseline parameters
    hidden_dim = 64,
    out_dim = 64,
    gnn_epochs = 200,
    gnn_lr = 1e-3,
    gnn_weight_decay = 5e-4,
    train_anchor_ratio = 0.2,
    device = "cuda" if torch.cuda.is_available() else "cpu",
    runtime_warmup = True,
    runtime_warmup_epochs = 5,
)

DATASETS = [
     "Chameleon","Texas", "Cora", "CiteSeer", "Actor", "Amazon-Computers", "Amazon-Photo", "Chameleon", "Squirrel",
    "Cora", "CiteSeer", "PubMed",
    "Cornell",  "Wisconsin",
    "WikiCS",
]

P_FIXE_VALUES = np.round(np.linspace(0.0, 0.5, 11), 2)
NOISE_S_VALUES = np.round(np.linspace(0.0, 0.5, 11), 2)
NOISE_F_VALUES = [0.3]

GNN_MODELS = [
    "GCNConv", "SAGEConv", "GATConv", "GraphConv", "GINConv",
    "TAGConv", "ChebConv", "SGConv", "ARMAConv",
]

# ============================================================
# DATA
# ============================================================

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


def compute_beta(A):
    A_csr = A.tocsr()
    n = A_csr.shape[0]

    adj = [
        set(A_csr.indices[A_csr.indptr[i]:A_csr.indptr[i + 1]].tolist())
        for i in range(n)
    ]

    rows_e, cols_e = A_csr.nonzero()
    common_set, common_size = {}, {}

    for z, v in zip(rows_e, cols_e):
        if (z, v) in common_set:
            continue
        s = adj[z] & adj[v]
        common_set[(z, v)] = s
        common_set[(v, z)] = s
        common_size[(z, v)] = len(s)
        common_size[(v, z)] = len(s)

    rows, cols, vals = [], [], []

    for z, v in zip(rows_e, cols_e):
        sset = common_set[(z, v)]
        sz = len(sset)
        if sz == 0:
            continue

        acc = 0.0
        for u in sset:
            su = common_size.get((z, u), 0)
            denom = sz + su - 1
            if denom > 0:
                acc += (su - 1) / denom

        rows.append(z)
        cols.append(v)
        vals.append(acc / sz)

    return sp.coo_matrix((vals, (rows, cols)), shape=A.shape).tocsr()

def propagate(A, k=2, alpha=0.5, use_beta=True):
    n = A.shape[0]
    I = sp.eye(n, format="csr")
    M = I + alpha * (A.multiply(compute_beta(A)) if use_beta else A)
    W = I.copy()
    for _ in range(k):
        W = M @ W
        rs = np.asarray(W.sum(axis=1)).ravel()
        rs[rs == 0] = 1
        W = sp.diags(1 / rs) @ W
    W.eliminate_zeros()
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
        keep = np.argpartition(dat, -topk)[-topk:] if len(dat) > topk else np.arange(len(dat))
        rs.extend([i] * len(keep))
        cs.extend(idx[keep])
        vs.extend(dat[keep])
    Wk = sp.coo_matrix((vs, (rs, cs)), shape=W.shape).tocsr()
    nrm = np.asarray(Wk.sum(axis=1)).ravel()
    return (sp.diags(1 / np.maximum(nrm, 1e-12)) @ Wk).tocsr()

# ============================================================
# SIMILARITIES + FUSION
# ============================================================

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
    S_s = (_sparse_l2(W_A)[queries, :] @ _sparse_l2(W_B).T).toarray().astype(np.float32)
    S_f = (_row_l2(X_A)[queries, :] @ _row_l2(X_B).T).astype(np.float32)
    return S_s, S_f

def fuse_fixed(S_s, S_f, lam):
    return float(lam) * S_s + (1.0 - float(lam)) * S_f

def fuse_adaptive(S_s, S_f, topk=None, tau=10.0):
    def confidence(S):
        Z = tau * S - (tau * S).max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)
        H = -np.sum(P * np.log(P + 1e-12), axis=1)
        return 1.0 - H / np.log(S.shape[1])

    cs, cc = confidence(S_s), confidence(S_f)
    lam = cs / np.maximum(cs + cc, 1e-12)
    S = lam[:, None] * S_s + (1.0 - lam[:, None]) * S_f
    return S, lam

# ============================================================
# EVAL
# ============================================================

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

# ============================================================
# ORIGINAL RUN_ONE
# ============================================================

def run_one(data, p_replace, lam, edge_drop_prob, feat_noise_prob, cfg, seed,
            use_beta, fusion="fixed"):
    A1, A2, X_A, X_B, queries, targets, N = make_views(
        data, p_replace, edge_drop_prob, feat_noise_prob, seed=seed)
    if not len(queries):
        return {"acc": 0, "mrr": 0, "hits@5": 0, "hits@10": 0}, 0.0
    W_A = row_topk(propagate(A1, cfg["k"], cfg["alpha"], use_beta), cfg["topk"])
    W_B = row_topk(propagate(A2, cfg["k"], cfg["alpha"], use_beta), cfg["topk"])
    S_s, S_f = base_similarities(W_A, W_B, X_A, X_B, queries)
    if fusion == "adaptive":
        S, alpha = fuse_adaptive(S_s, S_f, cfg["topk"])
        return evaluate(S, targets), float(alpha.mean())
    return evaluate(fuse_fixed(S_s, S_f, lam), targets), float(lam)

# ============================================================
# GNN MODELS
# ============================================================

def scipy_to_edge_index(A):
    A = A.tocoo()
    edge_index = np.vstack([A.row, A.col]).astype(np.int64)
    return torch.tensor(edge_index, dtype=torch.long)

class GNNAlignmentEncoder(nn.Module):
    def __init__(self, model_name, in_dim, hidden_dim=64, out_dim=64, dropout=0.3):
        super().__init__()
        self.model_name = model_name
        self.dropout = dropout

        if model_name == "GCNConv":
            from torch_geometric.nn import GCNConv
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, out_dim)

        elif model_name == "SAGEConv":
            from torch_geometric.nn import SAGEConv
            self.conv1 = SAGEConv(in_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, out_dim)

        elif model_name == "GATConv":
            from torch_geometric.nn import GATConv
            self.conv1 = GATConv(in_dim, hidden_dim, heads=2, concat=False)
            self.conv2 = GATConv(hidden_dim, out_dim, heads=2, concat=False)

        elif model_name == "GraphConv":
            from torch_geometric.nn import GraphConv
            self.conv1 = GraphConv(in_dim, hidden_dim)
            self.conv2 = GraphConv(hidden_dim, out_dim)

        elif model_name == "GINConv":
            from torch_geometric.nn import GINConv
            mlp1 = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            mlp2 = nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))
            self.conv1 = GINConv(mlp1)
            self.conv2 = GINConv(mlp2)

        elif model_name == "TAGConv":
            from torch_geometric.nn import TAGConv
            self.conv1 = TAGConv(in_dim, hidden_dim, K=2)
            self.conv2 = TAGConv(hidden_dim, out_dim, K=2)

        elif model_name == "ChebConv":
            from torch_geometric.nn import ChebConv
            self.conv1 = ChebConv(in_dim, hidden_dim, K=2)
            self.conv2 = ChebConv(hidden_dim, out_dim, K=2)

        elif model_name == "SGConv":
            from torch_geometric.nn import SGConv
            # SGConv already performs K-hop propagation internally.
            self.conv1 = SGConv(in_dim, hidden_dim, K=2)
            self.conv2 = SGConv(hidden_dim, out_dim, K=1)

        elif model_name == "ARMAConv":
            from torch_geometric.nn import ARMAConv
            self.conv1 = ARMAConv(in_dim, hidden_dim, num_stacks=1, num_layers=2)
            self.conv2 = ARMAConv(hidden_dim, out_dim, num_stacks=1, num_layers=2)

        else:
            raise ValueError(f"Unknown GNN model: {model_name}")

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.normalize(x, p=2, dim=1)

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

def alignment_loss(z_a, z_b, queries, targets, train_idx, temperature=0.1):
    q = torch.tensor(queries[train_idx], dtype=torch.long, device=z_a.device)
    t = torch.tensor(targets[train_idx], dtype=torch.long, device=z_a.device)

    logits = (z_a[q] @ z_b.T) / temperature
    loss_ab = F.cross_entropy(logits, t)

    logits_ba = (z_b[t] @ z_a.T) / temperature
    loss_ba = F.cross_entropy(logits_ba, q)

    return 0.5 * (loss_ab + loss_ba)

def run_one_gnn(data, p_replace, edge_drop_prob, feat_noise_prob, cfg, seed, model_name):
    torch.manual_seed(seed)
    np.random.seed(seed)

    A1, A2, X_A, X_B, queries, targets, N = make_views(
        data,
        p_replace,
        edge_drop_prob,
        feat_noise_prob,
        seed=seed
    )

    if not len(queries):
        return {"acc": 0, "mrr": 0, "hits@5": 0, "hits@10": 0}

    train_idx, eval_idx = split_train_eval_pairs(
        queries,
        targets,
        train_anchor_ratio=cfg["train_anchor_ratio"],
        seed=seed
    )

    device = torch.device(cfg["device"])

    x_a = torch.tensor(X_A, dtype=torch.float32, device=device)
    x_b = torch.tensor(X_B, dtype=torch.float32, device=device)

    edge_a = scipy_to_edge_index(A1).to(device)
    edge_b = scipy_to_edge_index(A2).to(device)

    model = GNNAlignmentEncoder(
        model_name=model_name,
        in_dim=X_A.shape[1],
        hidden_dim=cfg["hidden_dim"],
        out_dim=cfg["out_dim"]
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["gnn_lr"],
        weight_decay=cfg["gnn_weight_decay"]
    )

    for _ in range(cfg["gnn_epochs"]):
        model.train()
        optimizer.zero_grad()
        z_a = model(x_a, edge_a)
        z_b = model(x_b, edge_b)
        loss = alignment_loss(z_a, z_b, queries, targets, train_idx)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        z_a = model(x_a, edge_a)
        z_b = model(x_b, edge_b)
        q_eval = torch.tensor(queries[eval_idx], dtype=torch.long, device=device)
        sim = (z_a[q_eval] @ z_b.T).detach().cpu().numpy()

    eval_targets = targets[eval_idx]
    return evaluate(sim, eval_targets)

# ============================================================
# FIGURE 1  (original method only)
# ============================================================

def experiment1(data, cfg, lam=1.0):
    print(f"\n=== Figure 1 : Accuracy vs p_replace (λ={lam}, no noise) ===")
    p_values = np.arange(0.0, 1.1, 0.1).round(2)
    out = {"with_beta": {}, "without_beta": {}}
    for p in p_values:
        for key, ub in [("with_beta", True), ("without_beta", False)]:
            accs = [run_one(data, p, lam, 0.0, 0.0, cfg, s, ub)[0]["acc"]
                    for s in range(cfg["n_seeds"])]
            out[key][p] = (float(np.mean(accs)), float(np.std(accs)))
        print(f"  p={p:.1f}  with_β={out['with_beta'][p][0]:.3f}  "
              f"without_β={out['without_beta'][p][0]:.3f}")
    return out, p_values

def plot_figure1(res, p_values, cfg, save_path, lam=0.5):
    _style()
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for key, label, color, mk, ls in [
        ("with_beta", r"With $\beta$ (structural dependence)",  "#2166ac", "o", "-"),
        ("without_beta", r"Without $\beta$ (uniform propagation)", "#d6604d", "s", "--"),
    ]:
        m = np.array([res[key][p][0] for p in p_values])
        e = np.array([res[key][p][1] for p in p_values])
        ax.plot(p_values, m, f"{mk}{ls}", color=color, label=label, lw=1.6, ms=4.5)
        ax.fill_between(p_values, m - e, m + e, alpha=0.12, color=color)
    ax.set_xlabel("Fraction of replaced nodes $p$", fontsize=9)
    ax.set_ylabel("Accuracy (Hits@1)", fontsize=9)
    ax.set_title(f"{cfg['dataset']} — Alignment accuracy vs. replacement ratio  "
                 f"($\\lambda={lam}$, no noise)", fontsize=8.5)
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

def synchronize_device(cfg):
    if cfg.get("device") == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()

def warmup_gnn_runtime(data, cfg, model_name, p_replace, noise_s, noise_f):
    if not cfg.get("runtime_warmup", True):
        return

    warmup_cfg = cfg.copy()
    warmup_cfg["gnn_epochs"] = int(cfg.get("runtime_warmup_epochs", 5))

    _ = run_one_gnn(
        data=data,
        p_replace=p_replace,
        edge_drop_prob=noise_s,
        feat_noise_prob=noise_f,
        cfg=warmup_cfg,
        seed=0,
        model_name=model_name
    )
    synchronize_device(cfg)

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    def safe(s):
        return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

    for ds in DATASETS:
        cfg = CONFIG.copy()
        cfg["dataset"] = ds

        tag = safe(ds)
        dataset_path = os.path.join(PATH, tag)
        os.makedirs(dataset_path, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"Dataset : {ds}")
        print("=" * 80)

        data = load_tg_dataset(ds)
        print(f"Dataset : {ds} | Nodes : {data.num_nodes} | Edges : {data.edge_index.shape[1] // 2}")

        # ------------------------------------------------------------
        # GNN-only experiment
        # ------------------------------------------------------------
        # In the GNN baselines, beta is not used. Each GNN layer performs
        # its own message passing on the ordinary edge_index of G1 and G2.
        # Therefore, we skip the original beta / no-beta DAWAK comparison here.

        # ------------------------------------------------------------
        # GNN baselines with same p/noise settings
        # ------------------------------------------------------------
        gnn_rows = []

        for model_name in GNN_MODELS:
            print("\n" + "-" * 80)
            print(f"Running GNN model: {model_name} on {ds}")
            print("-" * 80)

            model_path = os.path.join(dataset_path, model_name)
            os.makedirs(model_path, exist_ok=True)

            # Warm-up once per dataset/model so the measured runtime does not
            # include first-use overheads such as imports, CUDA initialization,
            # kernel warm-up, or initial memory allocation.
            warmup_gnn_runtime(
                data=data,
                cfg=cfg,
                model_name=model_name,
                p_replace=P_FIXE_VALUES[0],
                noise_s=NOISE_S_VALUES[0],
                noise_f=NOISE_F_VALUES[0],
            )

            for P_FIXE in P_FIXE_VALUES:
                for NOISE_S in NOISE_S_VALUES:
                    if not np.isclose(P_FIXE, NOISE_S):
                        continue
                    for NOISE_F in NOISE_F_VALUES:
                        print(f"\n--- {model_name} ({ds}) | S:{NOISE_S}, F:{NOISE_F}, p={P_FIXE} ---")

                        metrics_by_seed = []
                        times_by_seed = []
                        for s in range(cfg["n_seeds"]):
                            synchronize_device(cfg)
                            start_time = time.perf_counter()
                            r = run_one_gnn(
                                data=data,
                                p_replace=P_FIXE,
                                edge_drop_prob=NOISE_S,
                                feat_noise_prob=NOISE_F,
                                cfg=cfg,
                                seed=s,
                                model_name=model_name
                            )
                            synchronize_device(cfg)
                            elapsed_time = time.perf_counter() - start_time
                            metrics_by_seed.append(r)
                            times_by_seed.append(elapsed_time)

                        accs = np.array([r["acc"] for r in metrics_by_seed])
                        mrrs = np.array([r["mrr"] for r in metrics_by_seed])
                        h5s = np.array([r["hits@5"] for r in metrics_by_seed])
                        h10s = np.array([r["hits@10"] for r in metrics_by_seed])
                        times = np.array(times_by_seed)

                        row = {
                            "dataset": ds,
                            "model": model_name,
                            "p_replace": P_FIXE,
                            "noise_s": NOISE_S,
                            "noise_f": NOISE_F,
                            "acc_mean": float(accs.mean()),
                            "acc_std": float(accs.std()),
                            "mrr_mean": float(mrrs.mean()),
                            "mrr_std": float(mrrs.std()),
                            "hits5_mean": float(h5s.mean()),
                            "hits5_std": float(h5s.std()),
                            "hits10_mean": float(h10s.mean()),
                            "hits10_std": float(h10s.std()),
                            "time": float(times.mean()),
                            "time_std": float(times.std()),
                        }

                        gnn_rows.append(row)

                        print(
                            f"  acc={row['acc_mean']:.4f} ± {row['acc_std']:.4f} | "
                            f"mrr={row['mrr_mean']:.4f} | "
                            f"hits@5={row['hits5_mean']:.4f} | "
                            f"hits@10={row['hits10_mean']:.4f} | "
                            f"time={row['time']:.2f}s ± {row['time_std']:.2f}s"
                        )

            try:
                import pandas as pd
                df_model = pd.DataFrame([r for r in gnn_rows if r["model"] == model_name])
                model_csv_path = os.path.join(model_path, f"{tag}_{model_name}_summary_results.csv")
                df_model.to_csv(model_csv_path, index=False)
                print(f"\nSaved {model_name} summary CSV: {model_csv_path}")
            except ImportError:
                print(f"\nPandas is not installed. {model_name} summary CSV was not saved.")

        try:
            import pandas as pd

            # Detailed CSV: one row for each
            # model × p_replace × noise_s × noise_f combination.
            df_gnn = pd.DataFrame(gnn_rows)
            gnn_csv_path = os.path.join(dataset_path, f"{tag}_gnn_all_models_summary_results.csv")
            df_gnn.to_csv(gnn_csv_path, index=False)
            print(f"\nSaved all GNN detailed summary CSV: {gnn_csv_path}")

            # Compact CSV: one final row per model, averaged over all
            # p_replace / noise_s / noise_f combinations.
            # This is the table format requested:
            # model, hit@1, hit@5, hit@10
            df_compact = (
                df_gnn
                .groupby("model", as_index=False)
                .agg({
                    "acc_mean": "mean",
                    "hits5_mean": "mean",
                    "hits10_mean": "mean",
                    "time": "mean",
                    "time_std": "mean",
                })
                .rename(columns={
                    "acc_mean": "hit@1",
                    "hits5_mean": "hit@5",
                    "hits10_mean": "hit@10",
                })
            )

            # Keep the model order fixed as in GNN_MODELS.
            df_compact["model"] = pd.Categorical(
                df_compact["model"],
                categories=GNN_MODELS,
                ordered=True
            )
            df_compact = df_compact.sort_values("model").reset_index(drop=True)
            df_compact.index = df_compact.index + 1

            compact_csv_path = os.path.join(dataset_path, f"{tag}_gnn_compact_results.csv")
            df_compact.to_csv(compact_csv_path, index=True, index_label="")
            print(f"Saved compact GNN result CSV: {compact_csv_path}")

        except ImportError:
            print("\nPandas is not installed. All GNN summary CSVs were not saved.")
