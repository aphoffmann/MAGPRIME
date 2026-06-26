"""DAFGrad: direction-and-frequency gradiometry.

DAFGrad depends on optional backends (nsgt, sklearn's HDBSCAN); skip if they
are unavailable.
"""

import numpy as np
import pytest

DAFGrad = pytest.importorskip("magprime.algorithms.interference.DAFGrad")


def test_dafgrad_runs(scenario, metrics, report):
    sc = scenario(gains=(1.0, 0.4), amp_factor=5.0)

    DAFGrad.fs = 1
    rec = DAFGrad.clean(np.copy(sc.B))

    assert rec.shape[-1] == sc.ambient.shape[-1]
    assert np.isfinite(rec).all()
    if rec.shape == sc.ambient.shape:
        err_raw = metrics.rmse(sc.noisy, sc.ambient)
        err_clean = metrics.rmse(rec, sc.ambient)
        report(
            rmse_raw=err_raw,
            rmse_clean=err_clean,
            reduction=err_raw / err_clean,
        )
        assert err_clean < 0.1 * err_raw
