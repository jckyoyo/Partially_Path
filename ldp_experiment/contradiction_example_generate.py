"""Generators for planted LDP budget-swap counterexamples.

The legacy public function ``generate_k_path_swap_counterexample`` is kept
compatible: it still returns ``(edges, k, delta, info)``. The dataclass wrapper
below is an interface convenience only and does not change generation logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


WeightedEdge = tuple[str, str, int]


@dataclass(frozen=True)
class CounterexampleInstance:
    """Structured view of a generated counterexample instance."""

    edges: list[WeightedEdge]
    k: int
    delta: int
    info: dict[str, Any]

    def as_legacy_tuple(self) -> tuple[list[WeightedEdge], int, int, dict[str, Any]]:
        """Return the historical ``(edges, k, delta, info)`` tuple."""
        return self.edges, self.k, self.delta, self.info


def generate_k_path_swap_counterexample(
    k: int,
    slots_per_pair: int = 3,
    release_alts: int = 2,
    use_alts: int = 2,
    alpha: int = 2,
    M: int = 40,
) -> tuple[list[WeightedEdge], int, int, dict[str, Any]]:
    """
    Generate a directed weighted graph family where an LDP-style sequential
    algorithm is likely to consume budget on bad common nodes x_{p,i},
    while the optimal solution releases those nodes and uses good common
    nodes y_{p,i}.

    Parameters
    ----------
    k : int
        Number of required link-disjoint s-t paths. Must be at least 3.

    slots_per_pair : int
        Number of budget-swap slots in each blocker-saver pair.
        Larger value gives more candidate cycles.

    release_alts : int
        Number of alternative bypasses around each bad node x_{p,i}.
        This increases the number of nonnegative-weight negative-cost cycles.

    use_alts : int
        Number of cheap alternatives through each good node y_{p,i}.
        This increases the number of negative-weight positive-cost cycles.

    alpha : int
        Extra weight paid when P0 bypasses a bad node x_{p,i}.
        The release cycle roughly has weight +alpha and cost -1.

    M : int
        Weight of each expensive direct edge in saver paths.
        Should be much larger than the cheap alternative weight plus alpha.

    Returns
    -------
    edges : list[tuple[str, str, int]]
        Directed weighted edges.

    k : int
        Number of required paths.

    delta : int
        Common-node budget.

    info : dict
        Intended LDP pattern, intended optimal pattern, and theoretical data.
    """

    if k < 3:
        raise ValueError("k must be at least 3.")

    edges: list[WeightedEdge] = []

    def add(u, v, w):
        edges.append((u, v, w))

    g = (k - 1) // 2
    m = slots_per_pair
    total_slots = g * m

    # Common-node budget.
    # LDP spends it on all x_{p,i}.
    # Optimal spends it on all y_{p,i}.
    delta = total_slots

    # ------------------------------------------------------------
    # 1. Build base path P0.
    #
    # P0_bad goes through x_{p,i}.
    # P0_good bypasses x_{p,i} but still goes through y_{p,i}.
    # ------------------------------------------------------------

    P0_bad = ["s"]
    P0_good = ["s"]

    prev = "s"

    for p in range(1, g + 1):
        for i in range(1, m + 1):
            L = f"L_{p}_{i}"
            x = f"x_{p}_{i}"
            R = f"R_{p}_{i}"
            y = f"y_{p}_{i}"
            N = f"N_{p}_{i}"

            # Enter this slot.
            add(prev, L, 1)

            # Bad route through x: L -> x -> R.
            add(L, x, 1)
            add(x, R, 1)

            # Multiple bypass alternatives around x.
            # Each gives a release candidate cycle.
            for a in range(1, release_alts + 1):
                bnode = f"bypass_{p}_{i}_{a}"
                # Total bypass weight = 2 + alpha + (a-1).
                add(L, bnode, 1)
                add(bnode, R, 1 + alpha + (a - 1))

            # Both bad and good base paths pass y.
            add(R, y, 1)
            add(y, N, 1)

            P0_bad.extend([L, x, R, y, N])
            P0_good.extend([L, R, y, N])

            prev = N

    add(prev, "t", 1)
    P0_bad.append("t")
    P0_good.append("t")

    P0_bad_weight = 5 * total_slots + 1
    P0_good_weight = P0_bad_weight + alpha * total_slots

    # ------------------------------------------------------------
    # Choose path weights to enforce intended LDP order:
    #
    # P0_bad first.
    # Then all blockers B_p.
    # Then all savers D_p, but forced to use expensive edges
    # because the budget is already consumed by x nodes.
    # ------------------------------------------------------------

    blocker_gate = P0_bad_weight + 10
    blocker_unit = 1
    blocker_end = 1

    blocker_weight = blocker_gate + 2 * m * blocker_unit + blocker_end

    saver_gate = blocker_weight + 10
    cheap_alt_base = 4
    saver_end = 1

    saver_cheap_weight = saver_gate + m * cheap_alt_base + saver_end
    saver_expensive_weight = saver_gate + m * M + saver_end

    filler_gate = saver_expensive_weight + 1000

    blocker_paths = []
    saver_exp_paths = []
    saver_cheap_paths = []

    # ------------------------------------------------------------
    # 2. Build blocker paths B_p.
    #
    # B_p passes all x_{p,i}, making them common with P0_bad.
    # ------------------------------------------------------------

    for p in range(1, g + 1):
        path = ["s", f"B_{p}_0"]
        add("s", f"B_{p}_0", blocker_gate)

        prev = f"B_{p}_0"

        for i in range(1, m + 1):
            x = f"x_{p}_{i}"
            nxt = f"B_{p}_{i}"

            add(prev, x, blocker_unit)
            add(x, nxt, blocker_unit)

            path.extend([x, nxt])
            prev = nxt

        add(prev, "t", blocker_end)
        path.append("t")

        blocker_paths.append(path)

    # ------------------------------------------------------------
    # 3. Build saver paths D_p.
    #
    # D_p has two choices at each slot:
    #   expensive direct edge: prev -> next
    #   cheap alternative through y_{p,i}
    #
    # The cheap alternative creates a positive-cost negative-weight
    # candidate cycle after LDP chooses the expensive edge.
    # ------------------------------------------------------------

    for p in range(1, g + 1):
        exp_path = ["s", f"D_{p}_0"]
        cheap_path = ["s", f"D_{p}_0"]

        add("s", f"D_{p}_0", saver_gate)
        prev = f"D_{p}_0"

        for i in range(1, m + 1):
            y = f"y_{p}_{i}"
            nxt = f"D_{p}_{i}"

            # Expensive direct edge.
            add(prev, nxt, M)

            # Multiple cheap alternatives through y.
            # Each gives a use candidate cycle.
            for b in range(1, use_alts + 1):
                in_node = f"use_in_{p}_{i}_{b}"
                out_node = f"use_out_{p}_{i}_{b}"

                # Total cheap alternative weight = 4 + (b-1).
                add(prev, in_node, 1)
                add(in_node, y, 1 + (b - 1))
                add(y, out_node, 1)
                add(out_node, nxt, 1)

            exp_path.append(nxt)
            cheap_path.extend([y, nxt])

            prev = nxt

        add(prev, "t", saver_end)
        exp_path.append("t")
        cheap_path.append("t")

        saver_exp_paths.append(exp_path)
        saver_cheap_paths.append(cheap_path)

    # ------------------------------------------------------------
    # 4. Add filler paths if k is even.
    #
    # These paths are disjoint and very expensive, so they do not
    # participate in the counterexample structure.
    # ------------------------------------------------------------

    filler_paths = []
    used_paths = 1 + 2 * g
    filler_count = k - used_paths

    for h in range(1, filler_count + 1):
        path = ["s", f"F_{h}_0", f"F_{h}_1", "t"]
        add("s", f"F_{h}_0", filler_gate + 10 * h)
        add(f"F_{h}_0", f"F_{h}_1", 1)
        add(f"F_{h}_1", "t", 1)
        filler_paths.append(path)

    # ------------------------------------------------------------
    # 5. Theoretical comparison.
    # ------------------------------------------------------------

    greedy_weight = (
        P0_bad_weight
        + g * blocker_weight
        + g * saver_expensive_weight
        + sum(filler_gate + 10 * h + 2 for h in range(1, filler_count + 1))
    )

    optimal_pattern_weight = (
        P0_good_weight
        + g * blocker_weight
        + g * saver_cheap_weight
        + sum(filler_gate + 10 * h + 2 for h in range(1, filler_count + 1))
    )

    # Per slot:
    # release x costs about alpha,
    # use y saves about M - cheap_alt_base.
    per_slot_improvement = M - cheap_alt_base - alpha

    info = {
        "g_pairs": g,
        "slots_per_pair": m,
        "total_slots": total_slots,
        "delta": delta,
        "P0_bad": P0_bad,
        "P0_good": P0_good,
        "blocker_paths": blocker_paths,
        "saver_expensive_paths": saver_exp_paths,
        "saver_cheap_paths": saver_cheap_paths,
        "filler_paths": filler_paths,
        "P0_bad_weight": P0_bad_weight,
        "P0_good_weight": P0_good_weight,
        "blocker_weight_each": blocker_weight,
        "saver_expensive_weight_each": saver_expensive_weight,
        "saver_cheap_weight_each": saver_cheap_weight,
        "greedy_pattern_weight": greedy_weight,
        "optimal_pattern_weight": optimal_pattern_weight,
        "expected_improvement": greedy_weight - optimal_pattern_weight,
        "per_slot_improvement": per_slot_improvement,
        "candidate_release_cycles_at_least": total_slots * release_alts,
        "candidate_use_cycles_at_least": total_slots * use_alts,
        "candidate_cycles_at_least": total_slots * (release_alts + use_alts),
        "bad_common_nodes": [f"x_{p}_{i}" for p in range(1, g + 1) for i in range(1, m + 1)],
        "good_common_nodes": [f"y_{p}_{i}" for p in range(1, g + 1) for i in range(1, m + 1)],
    }

    return edges, k, delta, info


def generate_counterexample_instance(
    k: int,
    slots_per_pair: int = 3,
    release_alts: int = 2,
    use_alts: int = 2,
    alpha: int = 2,
    M: int = 40,
) -> CounterexampleInstance:
    """Generate the same instance as a dataclass instead of a raw tuple."""
    edges, k_out, delta, info = generate_k_path_swap_counterexample(
        k=k,
        slots_per_pair=slots_per_pair,
        release_alts=release_alts,
        use_alts=use_alts,
        alpha=alpha,
        M=M,
    )
    return CounterexampleInstance(edges=edges, k=k_out, delta=delta, info=info)
