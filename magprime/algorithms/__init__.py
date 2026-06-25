# magprime/algorithms/__init__.py
#
# Algorithms are exposed lazily and defensively: several of them depend on
# heavy, optional third-party backends (e.g. pymssa, cvxpy, hdbscan, nsgt,
# invertiblewavelets).  Importing magprime.algorithms should never hard-fail
# just because one optional backend is not installed -- the algorithms whose
# dependencies are unavailable are simply omitted from the namespace.

import importlib as _importlib

# (subpackage -> module names) for every shipped algorithm.
_ALGORITHMS = {
    "interference": [
        "DAFGrad", "ICA", "MSSA", "NESS", "NEUBAUER", "PiCoG",
        "RAMEN", "REAM", "SHEINKER", "UBSS", "WAICUP",
    ],
    "anomaly": ["RUDE", "RUDER"],
    "spectral": ["AnomalyTracker", "LikelihoodRatio"],
    "survey": ["PFSS"],
}

__all__ = []

for _subpkg, _names in _ALGORITHMS.items():
    for _name in _names:
        try:
            _module = _importlib.import_module(f".{_subpkg}.{_name}", __name__)
        except ImportError:
            # Optional dependency for this algorithm is not installed; skip it.
            continue
        globals()[_name] = _module
        __all__.append(_name)

del _importlib, _subpkg, _names, _name
