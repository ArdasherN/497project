"""
dataset.py  (single-topology, many-traffic-matrices)

Goal: one fixed topology + N traffic matrices for ML experiments.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple, Optional

import networkx as nx
import numpy as np
import pandas as pd

# Helpers

def _read_positions(G: nx.Graph) -> Dict[str, Tuple[float, float]]:
    # try GraphML coords, else KK layout
    pos = {}
    for n, d in G.nodes(data=True):
        if 'x' in d and 'y' in d:
            try:
                pos[n] = (float(d['x']), float(d['y']))
            except (TypeError, ValueError):
                pass
    if len(pos) != G.number_of_nodes():
        pos = nx.kamada_kawai_layout(G)
    return pos


def _directed_copy(G_und: nx.Graph) -> nx.DiGraph:
    # make edges bidirectional
    G = nx.DiGraph()
    for u, v, d in G_und.edges(data=True):
        G.add_edge(u, v, **d)
        G.add_edge(v, u, **d)
    return G


def _gravity_matrix(nodes: List[str], total: float, rng: np.random.Generator) -> pd.DataFrame:
    # simple gravity-model TM
    w = rng.random(len(nodes))
    D = np.outer(w, w)
    D = D / D.sum() * total
    np.fill_diagonal(D, 0)
    return pd.DataFrame(D.round(1), index=nodes, columns=nodes)

# Main loader

def load_single_topology(
    base_dir: str = '.',
    topo_name: str = 'Abilene',
    capacity: float = 1000,
    n_samples: int = 100,
    total_demand: float = 10_000,
    seed: Optional[int] = 42
):
    # load graph, set capacities, get positions, gen TMs
    src = os.path.join(base_dir, 'topologyzoo', 'sources', f'{topo_name}.graphml')
    if not os.path.isfile(src):
        raise FileNotFoundError(f"GraphML '{src}' not found")

    G_und = nx.read_graphml(src)
    G = _directed_copy(G_und)
    nx.set_edge_attributes(G, capacity, 'capacity')

    pos = _read_positions(G_und)
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())

    tms = [_gravity_matrix(nodes, total_demand, rng) for _ in range(n_samples)]
    return G, pos, tms

if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--zoo-path', default='.')
    p.add_argument('--topo', default='Abilene')
    p.add_argument('-n', type=int, default=5)
    args = p.parse_args()

    G, pos, mats = load_single_topology(args.zoo_path, args.topo, n_samples=args.n)
    print(f"Loaded '{args.topo}': {G.number_of_nodes()} nodes, {len(mats)} TMs")
    print(mats[0].iloc[:5, :5])
