"""WAICUP: wavelet-adaptive interference cancellation.

Requires the optional ``invertiblewavelets`` backend; the whole module is
skipped when it is not installed.
"""

import numpy as np
import pytest

WAICUP = pytest.importorskip("magprime.algorithms.interference.WAICUP")


def test_waicup_reduces_interference(scenario, metrics, report):
    sc = scenario(gains=(1.0, 0.4), amp_factor=5.0)

    WAICUP.fs = 1
    rec = WAICUP.clean(np.copy(sc.B))

    assert rec.shape == sc.ambient.shape
    assert np.isfinite(rec).all()

    err_raw = metrics.rmse(sc.noisy, sc.ambient)
    err_clean = metrics.rmse(rec, sc.ambient)
    report(rmse_raw=err_raw, rmse_clean=err_clean, reduction=err_raw / err_clean)
    assert err_clean < err_raw
