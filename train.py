"""
train.py – compare GCN vs GFNO on one graph + many TMs
"""
from __future__ import annotations
import argparse, random, os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx

from dataset import load_single_topology
from gcn import LinkUtilGCN, normalize_adj
from gfno import LinkUtilGFNO, laplacian_basis

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# node features [out, in, x, y]

def node_features(tm: pd.DataFrame, pos: dict) -> torch.Tensor:
    nodes = list(tm.index)
    out_sum = tm.sum(axis=1).values
    in_sum = tm.sum(axis=0).values
    coords = np.array([pos[n] for n in nodes])
    feat = np.concatenate([out_sum[:, None], in_sum[:, None], coords], axis=1)
    feat = feat / (feat.max() + 1e-9)
    return torch.tensor(feat, dtype=torch.float32)

# build X, Â, edges for each TM

def build_samples(G: nx.DiGraph,
                  pos: dict,
                  tms: List[pd.DataFrame]
                 ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                            List[str],
                            torch.Tensor]:
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    edge_index = torch.tensor([[idx[u], idx[v]] for u, v in G.edges()], dtype=torch.long)

    A = torch.tensor(nx.to_numpy_array(G, nodelist=nodes), dtype=torch.float32)
    A_hat = normalize_adj(A)

    samples = []
    for tm in tms:
        tm = tm.loc[nodes, nodes]  # align order
        X = node_features(tm, pos)
        samples.append((X, A_hat, edge_index))
    return samples, nodes, edge_index

# read util CSVs into tensors

def read_labels(topo: str,
                nodes: List[str],
                edge_index: torch.Tensor,
                n_samples: int) -> List[torch.Tensor]:
    labels = []
    for i in range(n_samples):
        path = os.path.join("results", f"{topo}_tm{i:03d}_util.csv")
        df = pd.read_csv(path)
        m = {(str(r.src), str(r.dst)): r.utilization for r in df.itertuples()}
        vec = [m[(nodes[u], nodes[v])] for u, v in edge_index.tolist()]
        labels.append(torch.tensor(vec, dtype=torch.float32))
    return labels

# one pass over data

def run_epoch(model, data, y_list, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    mse = nn.MSELoss()
    tot_l = tot_m = 0.0
    for (X, A_hat, ei), y in zip(data, y_list):
        X, y, ei = X.to(DEVICE), y.to(DEVICE), ei.to(DEVICE)
        pred = model(X, A_hat.to(DEVICE), ei) if isinstance(model, LinkUtilGCN) else model(X, ei)
        loss = mse(pred, y)
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        tot_l += loss.item()
        tot_m += (pred.detach() - y).abs().mean().item()
    n = len(data)
    return tot_l / n, tot_m / n

# main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zoo-path", default=".")
    p.add_argument("--topo", default="Abilene")
    p.add_argument("-n", type=int, default=100)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    # data
    G, pos, tms = load_single_topology(args.zoo_path, args.topo, n_samples=args.n)
    samples, nodes, edge_index = build_samples(G, pos, tms)
    labels = read_labels(args.topo, nodes, edge_index, args.n)

    # split 70/15/15
    idxs = list(range(args.n)); random.shuffle(idxs)
    n_tr = int(0.7 * args.n); n_val = int(0.15 * args.n)
    tr_idx, val_idx, te_idx = idxs[:n_tr], idxs[n_tr:n_tr + n_val], idxs[n_tr + n_val:]

    tr_data = [samples[i] for i in tr_idx]; tr_y = [labels[i] for i in tr_idx]
    val_data = [samples[i] for i in val_idx]; val_y = [labels[i] for i in val_idx]
    te_data = [samples[i] for i in te_idx]; te_y = [labels[i] for i in te_idx]

    # models
    gcn = LinkUtilGCN().to(DEVICE)

    A = torch.tensor(nx.to_numpy_array(G, nodelist=nodes), dtype=torch.float32)
    U, _ = laplacian_basis(A, K=20)
    gfno = LinkUtilGFNO(U, hidden=64).to(DEVICE)

    results = {}
    for name, model in [("GCN", gcn), ("GFNO", gfno)]:
        opt = optim.Adam(model.parameters(), lr=args.lr)
        best_mae, best_state = float('inf'), None
        for ep in range(1, args.epochs + 1):
            tr_l, tr_m = run_epoch(model, tr_data, tr_y, opt)
            val_l, val_m = run_epoch(model, val_data, val_y)
            if val_m < best_mae:
                best_mae, best_state = val_m, model.state_dict()
            if ep == 1 or ep % 10 == 0 or ep == args.epochs:
                print(f"{name} Ep {ep:3d} | train MAE {tr_m:.3f}  val MAE {val_m:.3f}")
        model.load_state_dict(best_state)
        te_l, te_m = run_epoch(model, te_data, te_y)
        results[name] = (te_l, te_m)
        print(f"{name} TEST | MAE {te_m:.4f}  MSE {te_l:.4f}\n")

    print("=== Summary ===")
    for n, (l, m) in results.items():
        print(f"{n}: MAE {m:.4f} | MSE {l:.4f}")

if __name__ == "__main__":
    main()
