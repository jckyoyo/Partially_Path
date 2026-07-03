"""Compatibility wrapper for the Lagrangian B&B implementation.

The implementation lives under ``ldp_experiment/B&B``. That directory name is
kept for project organization, but it is not a valid Python package name, so
this module loads and re-exports the implementation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType


def _load_impl() -> ModuleType:
    path = Path(__file__).resolve().parent / "B&B" / "lagrangian_bnb.py"
    spec = importlib.util.spec_from_file_location("_ldp_lagrangian_bnb_impl", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load lagrangian_bnb implementation from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_impl = _load_impl()

BNBNode = _impl.BNBNode
BNBResult = _impl.BNBResult
solve_candidate_edge_lagrangian_bnb = _impl.solve_candidate_edge_lagrangian_bnb
solve_candidate_edge_manual_lagrangian_bnb = _impl.solve_candidate_edge_manual_lagrangian_bnb

__all__ = [
    "BNBNode",
    "BNBResult",
    "solve_candidate_edge_lagrangian_bnb",
    "solve_candidate_edge_manual_lagrangian_bnb",
]
