"""Shared fixtures for the MAGPRIME test-suite.

Every scenario loads *real* Swarm magnetometer residual data through
``magprime.utility.data_loader.load_swarm_data`` and treats it as the
"ambient" field that each algorithm is supposed to recover.  Artificial
spacecraft interference is then layered on top with a *known* per-sensor
coupling gain, so the tests can verify that the algorithms actually remove the
interference (and not just that they run).

The scenario builder mirrors the gradiometry assumption shared by the
interference algorithms: the ambient field is common-mode across all sensors,
while a single interference source couples into each sensor with a different
gain (an inboard sensor sees more interference than an outboard one).
"""

import types

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SWARM_START = 160_000   # sample offset into the bundled Swarm record
N_SAMPLES = 4_000       # samples per scenario (>1920 so REAM's rolling window fits)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def rmse(a, b):
    """Root-mean-square error between two arrays."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def corr(a, b):
    """Pearson correlation between two arrays (flattened)."""
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    return float(np.corrcoef(a, b)[0, 1])


@pytest.fixture(scope="session")
def metrics():
    return types.SimpleNamespace(rmse=rmse, corr=corr)


@pytest.fixture
def report(request):
    """Print a labelled, uniform metrics line for a test.

    Captured by pytest unless run with ``-s`` (or on failure).  Example:
        report(rmse_raw=err_raw, rmse_clean=err_clean, reduction=err_raw / err_clean)
    """
    def _report(**values):
        formatted = "  ".join(
            f"{k}={v:.4g}" if isinstance(v, (int, float)) else f"{k}={v}"
            for k, v in values.items()
        )
        print(f"\n[{request.node.name}] {formatted}")

    return _report


# ---------------------------------------------------------------------------
# Data fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def ambient():
    """Real Swarm residual field, shape (3, N_SAMPLES), used as ground truth."""
    data_loader = pytest.importorskip("magprime.utility.data_loader")
    B = np.asarray(data_loader.load_swarm_data(SWARM_START, SWARM_START + N_SAMPLES), float)
    assert B.shape == (3, N_SAMPLES), f"unexpected swarm data shape: {B.shape}"
    return B


@pytest.fixture
def rng():
    return np.random.default_rng(2025)


def _interference_source(n_samples, rng, n_tones=2, fmin=0.01, fmax=0.05):
    """Synthetic narrowband interference (sum of sinusoids), shape (3, n_samples)."""
    t = np.arange(n_samples)
    src = np.zeros((3, n_samples))
    for axis in range(3):
        for _ in range(n_tones):
            f = rng.uniform(fmin, fmax)
            phase = rng.uniform(0, 2 * np.pi)
            src[axis] += np.sin(2 * np.pi * f * t + phase)
    return src


@pytest.fixture
def scenario(ambient, rng):
    """Factory that builds a multi-sensor gradiometer scenario.

    ``make(...)`` returns a ``SimpleNamespace`` with:
        B            : (n_sensors, 3, N) sensor measurements
        ambient      : (3, N) the true ambient field (recovery target)
        gains        : per-sensor interference coupling gains
        interference : (3, N) the (scaled) interference added to sensor 0
        noisy        : (3, N) the noisiest single sensor, B[0], for a baseline

    Parameters
    ----------
    gains : sequence of float
        Interference coupling per sensor; ``len(gains)`` sets the sensor count.
        The first sensor is the "inboard" (noisiest) one.
    amp_factor : float
        Interference amplitude as a multiple of the ambient field std.
    n_tones : int
        Number of sinusoidal tones per axis.
    transient : (lo, hi) or None
        If given, the interference is zero outside the sample window [lo, hi)
        (used to exercise burst-detection algorithms such as REAM).
    """
    amb = ambient
    N = amb.shape[1]

    def make(gains=(1.0, 0.4), amp_factor=5.0, n_tones=2, transient=None):
        src = _interference_source(N, rng, n_tones=n_tones)
        if transient is not None:
            lo, hi = transient
            window = np.zeros(N)
            window[lo:hi] = 1.0
            src = src * window
        interference = amp_factor * float(np.std(amb)) * src
        B = np.empty((len(gains), 3, N))
        for i, g in enumerate(gains):
            B[i] = amb + g * interference
        return types.SimpleNamespace(
            B=B,
            ambient=amb,
            gains=tuple(gains),
            interference=interference,
            noisy=B[0],
        )

    return make
