"""
gfno.py – naive GFNO on fixed graph
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from typing import Tuple

# compute Laplacian eigenbasis
def laplacian_basis(A: torch.Tensor, K: int | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    # normalized Laplacian L, then eigendecompose
    N = A.size(0)
    deg = A.sum(1)
    D_inv_sqrt = torch.diag(torch.where(deg>0, deg.pow(-0.5), deg))
    L = torch.eye(N, device=A.device) - D_inv_sqrt @ A @ D_inv_sqrt
    lam, U = torch.linalg.eigh(L)
    # select or pad to K modes
    K = N if K is None or K> N else K
    if K < N:
        U, lam = U[:, :K], lam[:K]
    elif K > N:
        pad_u = torch.zeros(N, K-N, device=A.device)
        pad_l = torch.zeros(K-N, device=A.device)
        U = torch.cat([U, pad_u], dim=1)
        lam = torch.cat([lam, pad_l])
    return U, lam

# per-frequency diagonal filter
class SpectralConv(nn.Module):
    def __init__(self, Cin: int, Cout: int, K: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(Cin, Cout, K))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, X_hat: torch.Tensor) -> torch.Tensor:
        # X_hat: K x Cin -> K x Cout
        W = self.weight.permute(2,0,1)  # K,Cin,Cout
        return torch.einsum('kco,kc->ko', W, X_hat)

# two-layer spectral encoder
class GFNOEncoder(nn.Module):
    def __init__(self, Cin=4, hidden=64, K=20, dropout=0.2, U=None):
        super().__init__()
        if U is None:
            raise ValueError("Need precomputed U")
        self.register_buffer('U', U)      # N x K
        self.conv1 = SpectralConv(Cin, hidden, K)
        self.conv2 = SpectralConv(hidden, hidden, K)
        self.drop = dropout

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # project to spectral domain
        X_hat = self.U.T @ X            # K x Cin
        Y_hat = F.relu(self.conv1(X_hat))
        Y_hat = F.dropout(Y_hat, p=self.drop, training=self.training)
        Z_hat = self.conv2(Y_hat)      # K x hidden
        return self.U @ Z_hat           # N x hidden

# full GFNO model
class LinkUtilGFNO(nn.Module):
    def __init__(self, U: torch.Tensor, hidden=64):
        super().__init__()
        K = U.size(1)
        self.encoder = GFNOEncoder(4, hidden, K, dropout=0.2, U=U)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        H = self.encoder(X)            # N x hidden
        h_u = H[edge_index[:,0]]
        h_v = H[edge_index[:,1]]
        return self.edge_mlp(torch.cat([h_u, h_v], dim=1)).squeeze(1)

# builder helper
def build_gfno(G: nx.DiGraph, K=20, hidden=64) -> Tuple[LinkUtilGFNO, torch.Tensor, torch.Tensor, torch.Tensor]:
    # prepare basis, adj, and model
    A = torch.tensor(nx.to_numpy_array(G), dtype=torch.float32)
    U, _ = laplacian_basis(A, K)
    edge_index = torch.tensor(list(G.edges()), dtype=torch.long)
    model = LinkUtilGFNO(U, hidden)
    return model, U, A, edge_index

# quick test
if __name__ == "__main__":
    G = nx.karate_club_graph().to_directed()
    model, U, A, edge_idx = build_gfno(G, K=16, hidden=32)
    X = torch.randn(G.number_of_nodes(), 4)
    out = model(X, edge_idx)
    print("Edge preds:", out.shape)  # (E,)
