"""UBSS: underdetermined blind source separation (smoke test).

Requires the optional ``cvxpy`` / ``hdbscan`` backends; the whole module is
skipped when they are not installed.  UBSS is iterative and heavy, so this is
a smoke test of the basic 3-sensor pipeline.
"""

import numpy as np
import pytest

UBSS = pytest.importorskip("magprime.algorithms.interference.UBSS")


def test_ubss_runs(scenario, metrics, report):
    sc = scenario(gains=(1.0, 0.6, 0.3), amp_factor=5.0)

    UBSS.fs = 1
    UBSS.setMagnetometers(3)
    rec = UBSS.clean(np.copy(sc.B))

    assert rec.shape == sc.ambient.shape
    assert np.isfinite(rec).all()
    report(
        rmse_raw=metrics.rmse(sc.noisy, sc.ambient),
        rmse_clean=metrics.rmse(rec, sc.ambient),
    )
