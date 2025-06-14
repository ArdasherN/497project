"""
gcn.py – raw 2-layer GCN with degree normalization and edge MLP
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

# normalize A + I
def normalize_adj(A: torch.Tensor) -> torch.Tensor:
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A, dtype=torch.float32)
    A = A.clone()
    idx = torch.arange(A.size(0))
    A[idx, idx] = 1.0
    deg = A.sum(1)
    inv = deg.pow(-0.5)
    inv[torch.isinf(inv)] = 0
    D = torch.diag(inv)
    return D @ A @ D

# single graph conv layer
class GraphConv(nn.Module):
    def __init__(self, in_f: int, out_f: int, bias=True):
        super().__init__()
        self.W = nn.Parameter(torch.empty(in_f, out_f))
        self.b = nn.Parameter(torch.empty(out_f)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        if self.b is not None:
            nn.init.zeros_(self.b)

    def forward(self, X, H_hat):
        out = H_hat @ (X @ self.W)
        return out + (self.b if self.b is not None else 0)

# two-layer GCN encoder
class GCNEncoder(nn.Module):
    def __init__(self, in_f=4, hidden=64, dropout=0.3):
        super().__init__()
        self.gc1 = GraphConv(in_f, hidden)
        self.gc2 = GraphConv(hidden, hidden)
        self.drop = dropout

    def forward(self, X, H_hat):
        H = F.relu(self.gc1(X, H_hat))
        H = F.dropout(H, p=self.drop, training=self.training)
        return self.gc2(H, H_hat)

# full link util model
class LinkUtilGCN(nn.Module):
    def __init__(self, in_feats=4, hidden=64, dropout=0.3):
        super().__init__()
        self.encoder = GCNEncoder(in_feats, hidden, dropout)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, X: torch.Tensor, H_hat: torch.Tensor, edge_index: torch.Tensor):
        H = self.encoder(X, H_hat)
        h_u = H[edge_index[:, 0]]
        h_v = H[edge_index[:, 1]]
        return self.edge_mlp(torch.cat([h_u, h_v], dim=1)).squeeze(1)

# quick test
if __name__ == "__main__":
    G = nx.karate_club_graph().to_directed()
    A = torch.tensor(nx.to_numpy_array(G), dtype=torch.float32)
    H_hat = normalize_adj(A)
    X = torch.randn(G.number_of_nodes(), 4)
    edge_idx = torch.tensor(list(G.edges())[:30], dtype=torch.long)

    model = LinkUtilGCN()
    out = model(X, H_hat, edge_idx)
    print("Pred shape:", out.shape)  # (E,)
