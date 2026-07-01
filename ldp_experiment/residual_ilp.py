"""Gurobi ILP baselines for residual circulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time

import networkx as nx

from .candidate_cycles import Cycle
from .graph_utils import EPS, edge_by_id


@dataclass
class ILPResult:
    objective: float
    total_cost: int
    selected_edge_ids: list[int]
    improved: bool
    status: str
    runtime_sec: float


def _solve_edge_subset_ilp(
    R: nx.MultiDiGraph,
    B: int,
    allowed_eids: set[int],
    time_limit: Optional[float],
    mip_gap: Optional[float],
) -> ILPResult:
    start = time.perf_counter()
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return ILPResult(0.0, 0, [], False, "NO_GUROBI", time.perf_counter() - start)
    by_id = edge_by_id(R)
    edges = [e for eid, e in by_id.items() if eid in allowed_eids]
    model = gp.Model("residual_circulation")
    model.Params.OutputFlag = 0
    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)
    if mip_gap is not None:
        model.Params.MIPGap = float(mip_gap)
    x = {e.eid: model.addVar(vtype=GRB.BINARY, name=f"x_{e.eid}") for e in edges}
    model.setObjective(gp.quicksum(e.weight * x[e.eid] for e in edges), GRB.MINIMIZE)
    model.addConstr(gp.quicksum(e.cost * x[e.eid] for e in edges) <= B, name="budget")
    for v in R.nodes:
        out_expr = gp.quicksum(eid_var for eid, eid_var in x.items() if by_id[eid].u == v)
        in_expr = gp.quicksum(eid_var for eid, eid_var in x.items() if by_id[eid].v == v)
        model.addConstr(out_expr - in_expr == 0, name=f"flow_{v}")
    model.optimize()
    status = model.Status
    status_name = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.CUTOFF: "CUTOFF",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INPROGRESS: "INPROGRESS",
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
        GRB.WORK_LIMIT: "WORK_LIMIT",
        GRB.MEM_LIMIT: "MEM_LIMIT",
    }.get(status, str(status))
    if model.SolCount == 0:
        return ILPResult(0.0, 0, [], False, status_name, time.perf_counter() - start)
    selected = [eid for eid, var in x.items() if var.X > 0.5]
    objective = float(model.ObjVal)
    total_cost = int(round(sum(by_id[eid].cost for eid in selected)))
    return ILPResult(objective, total_cost, sorted(selected), objective < -EPS, status_name, time.perf_counter() - start)


def solve_residual_circulation_ilp(
    R: nx.MultiDiGraph,
    B: int,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
) -> ILPResult:
    """Solve the full residual edge-variable circulation ILP."""
    return _solve_edge_subset_ilp(R, B, set(edge_by_id(R)), time_limit, mip_gap)


def solve_candidate_edge_subgraph_ilp(
    R: nx.MultiDiGraph,
    cycles: list[Cycle],
    B: int,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
) -> ILPResult:
    """Solve the circulation ILP restricted to the union of candidate-cycle edges."""
    allowed: set[int] = set()
    for cycle in cycles:
        allowed.update(cycle.edge_ids)
    return _solve_edge_subset_ilp(R, B, allowed, time_limit, mip_gap)
