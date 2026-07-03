"""Candidate-edge-guided Lagrangian branch-and-bound.

The primal problem is the minimum-weight budget-feasible circulation on the
complete residual graph:

    min w^T x
    s.t. Ax = 0, c^T x <= B, x in {0, 1}.

The empty circulation is feasible, so the initial incumbent upper bound is 0.
Candidate key edges are used only to choose branch variables; no residual edge
is deleted from the model. The Lagrangian relaxation moves the budget
constraint into the objective:

    L(x, lambda) = w^T x + lambda * (c^T x - B), lambda >= 0.

For any nonnegative lambda, the optimal Lagrangian value is a valid lower bound
for the budgeted problem at the current branch node. A Lagrangian solution that
exceeds the budget still gives a valid lower bound, but it cannot improve the
incumbent. Only budget-feasible solutions update the upper bound, using their
true objective w^T x. If a node lower bound is at least the incumbent UB, the
node is safely pruned. Once all candidate key edges have been fixed, the solver
calls the exact full residual ILP with those fixed variables, which preserves
correctness even though branching was guided by only a subset of edges.

Static SCC pruning belongs to candidate-edge extraction: it removes key-edge
hints that cannot lie on any directed cycle in the residual graph. The optional
node-level SCC check below is a stronger dynamic pruning rule after branch
exclusions. It is used only by ``manual_lagrangian_bnb`` mode; the default
``gurobi_priority`` mode delegates branch-and-bound to Gurobi.

This implementation intentionally does not include cost-feasibility pruning.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Optional

import networkx as nx

from ldp_experiment.candidate_edges import extract_candidate_key_edges
from ldp_experiment.graph_utils import EPS, edge_by_id
from ldp_experiment.residual_ilp import (
    solve_residual_circulation_ilp_with_candidate_priority,
    solve_residual_circulation_ilp_with_fixed,
)


@dataclass
class BNBResult:
    objective: float
    total_cost: int
    selected_edge_ids: list[int]
    improved: bool
    status: str
    runtime_sec: float
    num_nodes: int
    num_pruned_by_bound: int
    num_pruned_by_scc: int
    num_infeasible_lr: int
    num_exact_tail_calls: int
    num_candidate_edges: int
    best_lb: float


@dataclass(frozen=True)
class BNBNode:
    force_one: frozenset[int]
    force_zero: frozenset[int]
    lambda_init: float
    depth: int


@dataclass(frozen=True)
class _LRResult:
    feasible: bool
    status: str
    lagrangian_obj: float
    selected_edge_ids: list[int]
    real_weight: float
    real_cost: int
    fractional: bool


def _status_name(gp_status: int, GRB: object) -> str:
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
    }.get(gp_status, str(gp_status))


def _solve_lagrangian_relaxation_gurobi(
    R: nx.MultiDiGraph,
    B: int,
    force_one: frozenset[int],
    force_zero: frozenset[int],
    lam: float,
    time_limit: Optional[float] = None,
) -> _LRResult:
    """Solve one full-graph Lagrangian circulation LP.

    The budget constraint is omitted. Variables are continuous in [0, 1]. The
    node-arc incidence matrix is totally unimodular, and fixed 0/1 bounds keep
    LP extreme points integral; ``fractional`` is still reported for diagnostics.
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return _LRResult(False, "NO_GUROBI", 0.0, [], 0.0, 0, False)

    by_id = edge_by_id(R)
    all_eids = set(by_id)
    if force_one & force_zero or not force_one <= all_eids or not force_zero <= all_eids:
        return _LRResult(False, "INFEASIBLE", 0.0, [], 0.0, 0, False)

    edges = list(by_id.values())
    model = gp.Model("lagrangian_residual_circulation")
    model.Params.OutputFlag = 0
    if time_limit is not None:
        model.Params.TimeLimit = max(float(time_limit), 0.0)

    x = {
        e.eid: model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"x_{e.eid}")
        for e in edges
    }
    model.setObjective(gp.quicksum((e.weight + lam * e.cost) * x[e.eid] for e in edges), GRB.MINIMIZE)
    for eid in force_one:
        model.addConstr(x[eid] == 1.0, name=f"force_one_{eid}")
    for eid in force_zero:
        model.addConstr(x[eid] == 0.0, name=f"force_zero_{eid}")
    for v in R.nodes:
        out_expr = gp.quicksum(var for eid, var in x.items() if by_id[eid].u == v)
        in_expr = gp.quicksum(var for eid, var in x.items() if by_id[eid].v == v)
        model.addConstr(out_expr - in_expr == 0.0, name=f"flow_{v}")

    model.optimize()
    status = _status_name(model.Status, GRB)
    if model.SolCount == 0:
        return _LRResult(False, status, 0.0, [], 0.0, 0, False)

    values = {eid: float(var.X) for eid, var in x.items()}
    selected = sorted(eid for eid, value in values.items() if value > 0.5)
    fractional = any(EPS < value < 1.0 - EPS for value in values.values())
    real_weight = float(sum(by_id[eid].weight for eid in selected))
    real_cost = int(round(sum(by_id[eid].cost for eid in selected)))
    lagrangian_obj = float(model.ObjVal - lam * B)
    return _LRResult(True, status, lagrangian_obj, selected, real_weight, real_cost, fractional)


class _LagrangianRelaxationModel:
    """Reusable full-graph Lagrangian circulation LP.

    The model owns all residual edge variables and flow-balance constraints.
    Each solve updates the objective coefficients and temporarily fixes bounds
    for the current B&B node. Bounds are restored before returning so one node's
    fixed variables cannot leak into another node.
    """

    def __init__(self, R: nx.MultiDiGraph) -> None:
        self.R = R
        self.by_id = edge_by_id(R)
        self.all_eids = set(self.by_id)
        self.edges = list(self.by_id.values())
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except Exception:
            self.available = False
            self.gp = None
            self.GRB = None
            self.model = None
            self.x = {}
            return

        self.available = True
        self.gp = gp
        self.GRB = GRB
        self.model = gp.Model("lagrangian_residual_circulation_cached")
        self.model.Params.OutputFlag = 0
        self.model.ModelSense = GRB.MINIMIZE
        self.x = {
            e.eid: self.model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"x_{e.eid}")
            for e in self.edges
        }
        self.model.update()
        for v in R.nodes:
            out_expr = gp.quicksum(var for eid, var in self.x.items() if self.by_id[eid].u == v)
            in_expr = gp.quicksum(var for eid, var in self.x.items() if self.by_id[eid].v == v)
            self.model.addConstr(out_expr - in_expr == 0.0, name=f"flow_{v}")

    def solve(
        self,
        B: int,
        force_one: frozenset[int],
        force_zero: frozenset[int],
        lam: float,
        time_limit: Optional[float] = None,
    ) -> _LRResult:
        if not self.available:
            return _LRResult(False, "NO_GUROBI", 0.0, [], 0.0, 0, False)

        if force_one & force_zero or not force_one <= self.all_eids or not force_zero <= self.all_eids:
            return _LRResult(False, "INFEASIBLE", 0.0, [], 0.0, 0, False)

        assert self.model is not None
        assert self.GRB is not None
        GRB = self.GRB
        model = self.model

        if time_limit is not None:
            model.Params.TimeLimit = max(float(time_limit), 0.0)

        for edge in self.edges:
            self.x[edge.eid].Obj = edge.weight + lam * edge.cost

        fixed_eids = set(force_one) | set(force_zero)
        old_bounds = {eid: (float(self.x[eid].LB), float(self.x[eid].UB)) for eid in fixed_eids}
        try:
            for eid in force_one:
                self.x[eid].LB = 1.0
                self.x[eid].UB = 1.0
            for eid in force_zero:
                self.x[eid].LB = 0.0
                self.x[eid].UB = 0.0
            model.optimize()
            status = _status_name(model.Status, GRB)
            if model.SolCount == 0:
                return _LRResult(False, status, 0.0, [], 0.0, 0, False)

            values = {eid: float(var.X) for eid, var in self.x.items()}
            selected = sorted(eid for eid, value in values.items() if value > 0.5)
            fractional = any(EPS < value < 1.0 - EPS for value in values.values())
            real_weight = float(sum(self.by_id[eid].weight for eid in selected))
            real_cost = int(round(sum(self.by_id[eid].cost for eid in selected)))
            lagrangian_obj = float(model.ObjVal - lam * B)
            return _LRResult(True, status, lagrangian_obj, selected, real_weight, real_cost, fractional)
        finally:
            for eid, (lb, ub) in old_bounds.items():
                self.x[eid].LB = lb
                self.x[eid].UB = ub


def _forced_edges_can_be_circulation(
    R: nx.MultiDiGraph,
    force_one: frozenset[int],
    force_zero: frozenset[int],
    by_id: Optional[dict[int, object]] = None,
) -> bool:
    by_id = edge_by_id(R) if by_id is None else by_id
    G = nx.DiGraph()
    G.add_nodes_from(R.nodes)
    for edge in by_id.values():
        if edge.eid not in force_zero:
            G.add_edge(edge.u, edge.v)
    scc_id = {}
    for sid, comp in enumerate(nx.strongly_connected_components(G)):
        for node in comp:
            scc_id[node] = sid
    return all(scc_id.get(by_id[eid].u) == scc_id.get(by_id[eid].v) for eid in force_one)


def _choose_branch_eid(
    by_id: dict[int, object],
    candidate_eids: set[int],
    force_one: frozenset[int],
    force_zero: frozenset[int],
    last_lr_selected: list[int],
) -> int:
    unfixed = candidate_eids - set(force_one) - set(force_zero)
    selected_candidates = [eid for eid in last_lr_selected if eid in unfixed]
    if selected_candidates:
        return min(selected_candidates, key=lambda eid: _branch_sort_key(by_id, eid))
    return min(unfixed, key=lambda eid: _branch_sort_key(by_id, eid))


def _branch_sort_key(by_id: dict[int, object], eid: int) -> tuple[int, int, float, int]:
    return (
        0 if by_id[eid].weight < -EPS else 1,
        0 if by_id[eid].cost < 0 else 1,
        -abs(by_id[eid].weight),
        eid,
    )


def _solve_manual_lagrangian_bnb(
    R: nx.MultiDiGraph,
    B: int,
    candidate_eids: Optional[set[int]] = None,
    max_nodes: Optional[int] = None,
    max_lagrangian_iters: int = 5,
    time_limit: Optional[float] = None,
    exact_tail_time_limit: Optional[float] = None,
    step_rule: str = "polyak",
    enable_dynamic_scc_pruning: bool = True,
) -> BNBResult:
    """Solve by the original hand-written Lagrangian B&B."""
    start = time.perf_counter()
    by_id = edge_by_id(R)
    candidate_set = extract_candidate_key_edges(R) if candidate_eids is None else set(candidate_eids)
    candidate_set &= set(by_id)
    lr_model = _LagrangianRelaxationModel(R)

    ub = 0.0
    best_solution: list[int] = []
    best_cost = 0
    best_lb = -math.inf
    nodes = 0
    pruned_by_bound = 0
    pruned_by_scc = 0
    infeasible_lr = 0
    exact_tail_calls = 0
    status = "OPTIMAL"
    stack = [BNBNode(frozenset(), frozenset(), 0.0, 0)]

    def elapsed() -> float:
        return time.perf_counter() - start

    def remaining_time(limit: Optional[float]) -> Optional[float]:
        if limit is None:
            return None
        return max(float(limit) - elapsed(), 0.0)

    def finish(result_status: str) -> BNBResult:
        lb = best_lb if best_lb > -math.inf else 0.0
        return BNBResult(
            objective=ub,
            total_cost=best_cost,
            selected_edge_ids=sorted(best_solution),
            improved=ub < -EPS,
            status=result_status,
            runtime_sec=elapsed(),
            num_nodes=nodes,
            num_pruned_by_bound=pruned_by_bound,
            num_pruned_by_scc=pruned_by_scc,
            num_infeasible_lr=infeasible_lr,
            num_exact_tail_calls=exact_tail_calls,
            num_candidate_edges=len(candidate_set),
            best_lb=lb,
        )

    while stack:
        if time_limit is not None and elapsed() >= time_limit:
            status = "TIME_LIMIT"
            break
        if max_nodes is not None and nodes >= max_nodes:
            status = "NODE_LIMIT"
            break

        node = stack.pop()
        nodes += 1

        if node.force_one & node.force_zero:
            continue
        if enable_dynamic_scc_pruning and not _forced_edges_can_be_circulation(R, node.force_one, node.force_zero, by_id):
            pruned_by_scc += 1
            continue

        lam = max(0.0, node.lambda_init)
        node_lb = -math.inf
        node_best_lambda = lam
        last_lr_selected: list[int] = []
        lr_failed = False
        iters = max(max_lagrangian_iters, 1)

        for it in range(iters):
            lr = lr_model.solve(
                B,
                node.force_one,
                node.force_zero,
                lam,
                time_limit=remaining_time(time_limit),
            )
            if lr.status == "NO_GUROBI":
                return finish("NO_GUROBI")
            if not lr.feasible:
                infeasible_lr += 1
                lr_failed = True
                break
            if lr.status != "OPTIMAL":
                return finish(f"LR_{lr.status}")

            last_lr_selected = lr.selected_edge_ids
            if lr.lagrangian_obj > node_lb:
                node_lb = lr.lagrangian_obj
                node_best_lambda = lam
            if not lr.fractional and lr.real_cost <= B and lr.real_weight < ub - EPS:
                ub = lr.real_weight
                best_solution = lr.selected_edge_ids
                best_cost = lr.real_cost

            g = lr.real_cost - B
            if step_rule == "polyak":
                alpha = max(ub - lr.lagrangian_obj, 0.0) / max(float(g * g), 1e-9)
            else:
                alpha = 1.0 / math.sqrt(it + 1.0)
            lam = max(0.0, lam + alpha * g)

        if lr_failed:
            continue
        if node_lb > best_lb:
            best_lb = node_lb
        if node_lb >= ub - EPS:
            pruned_by_bound += 1
            continue

        fixed_candidates = set(node.force_one) | set(node.force_zero)
        if candidate_set <= fixed_candidates:
            exact_tail_calls += 1
            tail_limit = exact_tail_time_limit
            if time_limit is not None:
                rem = remaining_time(time_limit)
                tail_limit = rem if tail_limit is None else min(tail_limit, rem)
            tail = solve_residual_circulation_ilp_with_fixed(
                R,
                B,
                set(node.force_one),
                set(node.force_zero),
                time_limit=tail_limit,
            )
            if tail.status == "NO_GUROBI":
                return finish("NO_GUROBI")
            if tail.status == "INFEASIBLE":
                continue
            if tail.selected_edge_ids and tail.total_cost <= B and tail.objective < ub - EPS:
                ub = tail.objective
                best_solution = tail.selected_edge_ids
                best_cost = tail.total_cost
            if tail.status != "OPTIMAL":
                status = f"TAIL_{tail.status}"
                break
            continue

        branch_eid = _choose_branch_eid(by_id, candidate_set, node.force_one, node.force_zero, last_lr_selected)
        include = BNBNode(
            force_one=frozenset(set(node.force_one) | {branch_eid}),
            force_zero=node.force_zero,
            lambda_init=node_best_lambda,
            depth=node.depth + 1,
        )
        exclude = BNBNode(
            force_one=node.force_one,
            force_zero=frozenset(set(node.force_zero) | {branch_eid}),
            lambda_init=node_best_lambda,
            depth=node.depth + 1,
        )
        stack.append(exclude)
        stack.append(include)

    return finish(status)


def solve_candidate_edge_lagrangian_bnb(
    R: nx.MultiDiGraph,
    B: int,
    candidate_eids: Optional[set[int]] = None,
    max_nodes: Optional[int] = None,
    max_lagrangian_iters: int = 5,
    time_limit: Optional[float] = None,
    exact_tail_time_limit: Optional[float] = None,
    step_rule: str = "polyak",
    enable_dynamic_scc_pruning: bool = True,
    solve_mode: str = "gurobi_priority",
    mip_gap: Optional[float] = None,
) -> BNBResult:
    """Solve the full residual circulation problem by the selected B&B mode.

    ``gurobi_priority`` is the recommended default: it solves the complete
    residual ILP once and uses candidate edges only as Gurobi BranchPriority
    guidance. ``manual_lagrangian_bnb`` keeps the original Python-level
    Lagrangian branch-and-bound for theory and comparison experiments.
    """
    candidate_set = extract_candidate_key_edges(R) if candidate_eids is None else set(candidate_eids)
    candidate_set &= set(edge_by_id(R))

    if solve_mode == "gurobi_priority":
        result = solve_residual_circulation_ilp_with_candidate_priority(
            R,
            B,
            candidate_eids=candidate_set,
            time_limit=time_limit,
            mip_gap=mip_gap,
        )
        node_count = -1 if result.node_count is None else int(round(result.node_count))
        best_lb = result.objective if result.status == "OPTIMAL" else 0.0
        return BNBResult(
            objective=result.objective,
            total_cost=result.total_cost,
            selected_edge_ids=result.selected_edge_ids,
            improved=result.improved,
            status=f"PRIORITY_ILP_{result.status}",
            runtime_sec=result.runtime_sec,
            num_nodes=node_count,
            num_pruned_by_bound=0,
            num_pruned_by_scc=0,
            num_infeasible_lr=0,
            num_exact_tail_calls=0,
            num_candidate_edges=len(candidate_set),
            best_lb=best_lb,
        )

    if solve_mode == "manual_lagrangian_bnb":
        return _solve_manual_lagrangian_bnb(
            R,
            B,
            candidate_eids=candidate_set,
            max_nodes=max_nodes,
            max_lagrangian_iters=max_lagrangian_iters,
            time_limit=time_limit,
            exact_tail_time_limit=exact_tail_time_limit,
            step_rule=step_rule,
            enable_dynamic_scc_pruning=enable_dynamic_scc_pruning,
        )

    raise ValueError(f"unknown solve_mode {solve_mode!r}")


def solve_candidate_edge_manual_lagrangian_bnb(
    R: nx.MultiDiGraph,
    B: int,
    candidate_eids: Optional[set[int]] = None,
    max_nodes: Optional[int] = None,
    max_lagrangian_iters: int = 5,
    time_limit: Optional[float] = None,
    exact_tail_time_limit: Optional[float] = None,
    step_rule: str = "polyak",
    enable_dynamic_scc_pruning: bool = True,
) -> BNBResult:
    """Public entry point for the hand-written Lagrangian B&B variant."""
    return _solve_manual_lagrangian_bnb(
        R,
        B,
        candidate_eids=candidate_eids,
        max_nodes=max_nodes,
        max_lagrangian_iters=max_lagrangian_iters,
        time_limit=time_limit,
        exact_tail_time_limit=exact_tail_time_limit,
        step_rule=step_rule,
        enable_dynamic_scc_pruning=enable_dynamic_scc_pruning,
    )
