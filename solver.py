"""
solver.py  (single-topology, multi-TM)

Solve min-max utilisation LP for each TM from dataset.load_single_topology.
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import networkx as nx
import pandas as pd
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, PULP_CBC_CMD

from dataset import load_single_topology

# LP solver core
def solve_minmax_te(G: nx.DiGraph, tm: pd.DataFrame) -> Tuple[float, Dict[Tuple[str, str], float]]:
    # setup problem
    prob = LpProblem("MinMaxTE", LpMinimize)
    t = LpVariable("t", lowBound=0)

    edges = list(G.edges())
    cap = nx.get_edge_attributes(G, "capacity")
    demands = [(s, d, tm.at[s, d]) for s in tm.index for d in tm.columns if tm.at[s, d] > 0]

    # flow vars for each demand-edge
    flows = {(s, d, i, j): LpVariable(f"f_{s}_{d}_{i}_{j}", lowBound=0)
             for s, d, _ in demands for i, j in edges}

    # capacity constraints
    for i, j in edges:
        prob += lpSum(flows[s, d, i, j] for s, d, _ in demands) <= cap[(i, j)] * t

    # flow conservation
    for s, d, vol in demands:
        for v in G.nodes():
            inflow = lpSum(flows[s, d, u, v] for u in G.predecessors(v))
            outflow = lpSum(flows[s, d, v, w] for w in G.successors(v))
            if v == s:
                prob += outflow - inflow == vol
            elif v == d:
                prob += inflow - outflow == vol
            else:
                prob += outflow - inflow == 0

    # objective & solve
    prob += t
    prob.solve(PULP_CBC_CMD(msg=False))

    # extract results
    t_opt = t.value()
    util = { (i, j): sum(flows[s, d, i, j].value() for s, d, _ in demands) / cap[(i, j)]
             for i, j in edges }
    return t_opt, util

# CLI runner
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--zoo-path', default='.')
    p.add_argument('--topo', default='Abilene')
    p.add_argument('-n', type=int, default=100)
    p.add_argument('--capacity', type=float, default=1000)
    p.add_argument('--total-demand', type=float, default=10000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output-dir', default='results')
    args = p.parse_args()

    # load data
    G, pos, tms = load_single_topology(args.zoo_path, args.topo,
                                        capacity=args.capacity,
                                        n_samples=args.n,
                                        total_demand=args.total_demand,
                                        seed=args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    summary = []

    # solve each TM
    for idx, tm in enumerate(tms):
        print(f"Solving TM{idx:03d}…", end=' ')
        t_opt, util = solve_minmax_te(G, tm)
        print(f"t*={t_opt:.3f}")

        # save per-link utils
        df = pd.DataFrame([(u, v, u_) for (u, v), u_ in util.items()],
                          columns=['src', 'dst', 'utilization'])
        df.to_csv(os.path.join(args.output_dir,
                    f"{args.topo}_tm{idx:03d}_util.csv"), index=False)

        summary.append((idx, t_opt))

    # save summary
    pd.DataFrame(summary, columns=['tm_id', 't_opt'])\
      .to_csv(os.path.join(args.output_dir, f'{args.topo}_summary.csv'), index=False)
    print(f"Done. Results in {args.output_dir}/")

if __name__ == '__main__':
    main()
