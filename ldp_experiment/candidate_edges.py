"""Compatibility wrapper for B&B candidate-edge utilities.

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
    path = Path(__file__).resolve().parent / "B&B" / "candidate_edges.py"
    spec = importlib.util.spec_from_file_location("_ldp_candidate_edges_impl", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load candidate_edges implementation from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_impl = _load_impl()

CandidateEdgeStats = _impl.CandidateEdgeStats
extract_candidate_key_edges = _impl.extract_candidate_key_edges
extract_candidate_key_edges_with_stats = _impl.extract_candidate_key_edges_with_stats

__all__ = [
    "CandidateEdgeStats",
    "extract_candidate_key_edges",
    "extract_candidate_key_edges_with_stats",
]
