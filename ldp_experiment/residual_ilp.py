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
    node_count: Optional[float] = None
    mip_gap: Optional[float] = None


def _gurobi_status_name(status: int, GRB: object) -> str:
    return {
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


def _model_node_count(model: object) -> Optional[float]:
    try:
        return float(model.NodeCount)
    except Exception:
        return None


def _model_mip_gap(model: object) -> Optional[float]:
    try:
        return float(model.MIPGap)
    except Exception:
        return None


def _solve_edge_subset_ilp(
    R: nx.MultiDiGraph,
    B: int,
    allowed_eids: set[int],
    time_limit: Optional[float],
    mip_gap: Optional[float],
    force_one: Optional[set[int]] = None,
    force_zero: Optional[set[int]] = None,
) -> ILPResult:
    start = time.perf_counter()
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return ILPResult(0.0, 0, [], False, "NO_GUROBI", time.perf_counter() - start)
    force_one = set() if force_one is None else set(force_one)
    force_zero = set() if force_zero is None else set(force_zero)
    if force_one & force_zero:
        return ILPResult(0.0, 0, [], False, "INFEASIBLE", time.perf_counter() - start)

    by_id = edge_by_id(R)
    if not force_one <= allowed_eids or not force_zero <= allowed_eids:
        return ILPResult(0.0, 0, [], False, "INFEASIBLE", time.perf_counter() - start)

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
    for eid in force_one:
        model.addConstr(x[eid] == 1, name=f"force_one_{eid}")
    for eid in force_zero:
        model.addConstr(x[eid] == 0, name=f"force_zero_{eid}")
    for v in R.nodes:
        out_expr = gp.quicksum(eid_var for eid, eid_var in x.items() if by_id[eid].u == v)
        in_expr = gp.quicksum(eid_var for eid, eid_var in x.items() if by_id[eid].v == v)
        model.addConstr(out_expr - in_expr == 0, name=f"flow_{v}")
    model.optimize()
    status_name = _gurobi_status_name(model.Status, GRB)
    if model.SolCount == 0:
        return ILPResult(
            0.0,
            0,
            [],
            False,
            status_name,
            time.perf_counter() - start,
            node_count=_model_node_count(model),
            mip_gap=_model_mip_gap(model),
        )
    selected = [eid for eid, var in x.items() if var.X > 0.5]
    objective = float(model.ObjVal)
    total_cost = int(round(sum(by_id[eid].cost for eid in selected)))
    return ILPResult(
        objective,
        total_cost,
        sorted(selected),
        objective < -EPS,
        status_name,
        time.perf_counter() - start,
        node_count=_model_node_count(model),
        mip_gap=_model_mip_gap(model),
    )


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


def solve_residual_circulation_ilp_with_fixed(
    R: nx.MultiDiGraph,
    B: int,
    force_one: set[int],
    force_zero: set[int],
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
) -> ILPResult:
    """Solve the full residual circulation ILP with fixed edge variables."""
    return _solve_edge_subset_ilp(
        R,
        B,
        set(edge_by_id(R)),
        time_limit,
        mip_gap,
        force_one=set(force_one),
        force_zero=set(force_zero),
    )


def solve_residual_circulation_ilp_with_candidate_priority(
    R: nx.MultiDiGraph,
    B: int,
    candidate_eids: Optional[set[int]] = None,
    force_one: Optional[set[int]] = None,
    force_zero: Optional[set[int]] = None,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    candidate_priority: int = 100,
    noncandidate_priority: int = 1,
    use_empty_mip_start: bool = True,
) -> ILPResult:
    """Solve the full residual ILP while prioritizing candidate edges in Gurobi.

    Candidate edges only guide Gurobi's internal branch-and-bound through
    ``BranchPriority``. Every residual edge remains in the model, so the result
    is the same exact optimization problem as ``solve_residual_circulation_ilp``.
    """
    start = time.perf_counter()
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return ILPResult(0.0, 0, [], False, "NO_GUROBI", time.perf_counter() - start)

    by_id = edge_by_id(R)
    all_eids = set(by_id)
    candidate_set = set() if candidate_eids is None else set(candidate_eids) & all_eids
    force_one_set = set() if force_one is None else set(force_one)
    force_zero_set = set() if force_zero is None else set(force_zero)
    if force_one_set & force_zero_set:
        return ILPResult(0.0, 0, [], False, "INFEASIBLE", time.perf_counter() - start)
    if not force_one_set <= all_eids or not force_zero_set <= all_eids:
        return ILPResult(0.0, 0, [], False, "INFEASIBLE", time.perf_counter() - start)

    edges = list(by_id.values())
    model = gp.Model("residual_circulation_candidate_priority")
    model.Params.OutputFlag = 0
    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)
    if mip_gap is not None:
        model.Params.MIPGap = float(mip_gap)

    x = {}
    for edge in edges:
        var = model.addVar(vtype=GRB.BINARY, name=f"x_{edge.eid}")
        x[edge.eid] = var
    model.update()
    for edge in edges:
        x[edge.eid].BranchPriority = candidate_priority if edge.eid in candidate_set else noncandidate_priority
        if use_empty_mip_start:
            x[edge.eid].Start = 0.0

    model.setObjective(gp.quicksum(edge.weight * x[edge.eid] for edge in edges), GRB.MINIMIZE)
    model.addConstr(gp.quicksum(edge.cost * x[edge.eid] for edge in edges) <= B, name="budget")
    for eid in force_one_set:
        model.addConstr(x[eid] == 1, name=f"force_one_{eid}")
    for eid in force_zero_set:
        model.addConstr(x[eid] == 0, name=f"force_zero_{eid}")
    for v in R.nodes:
        out_expr = gp.quicksum(var for eid, var in x.items() if by_id[eid].u == v)
        in_expr = gp.quicksum(var for eid, var in x.items() if by_id[eid].v == v)
        model.addConstr(out_expr - in_expr == 0, name=f"flow_{v}")

    model.optimize()
    status_name = _gurobi_status_name(model.Status, GRB)
    if model.SolCount == 0:
        return ILPResult(
            0.0,
            0,
            [],
            False,
            status_name,
            time.perf_counter() - start,
            node_count=_model_node_count(model),
            mip_gap=_model_mip_gap(model),
        )

    selected = [eid for eid, var in x.items() if var.X > 0.5]
    objective = float(model.ObjVal)
    total_cost = int(round(sum(by_id[eid].cost for eid in selected)))
    return ILPResult(
        objective,
        total_cost,
        sorted(selected),
        objective < -EPS,
        status_name,
        time.perf_counter() - start,
        node_count=_model_node_count(model),
        mip_gap=_model_mip_gap(model),
    )
