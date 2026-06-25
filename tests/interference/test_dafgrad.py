"""DAFGrad: direction-and-frequency gradiometry (smoke test).

DAFGrad depends on optional backends (nsgt, sklearn's HDBSCAN); skip if they
are unavailable.  This is a smoke test: it checks the method runs end-to-end
and returns a finite ambient estimate of the right length.
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
        report(
            rmse_raw=metrics.rmse(sc.noisy, sc.ambient),
            rmse_clean=metrics.rmse(rec, sc.ambient),
        )
