"""LikelihoodRatio: narrowband spectral-track detection.

We build a positive spectrogram (frequency x time) with a broadband background
and one strong narrowband track, then check the detector flags the track's
frequency bin and (almost) nothing else.
"""

import numpy as np

from magprime.algorithms.spectral import LikelihoodRatio


def test_likelihood_ratio_detects_track():
    rng = np.random.default_rng(0)
    n_freq, n_time = 64, 120

    s = 1.0 + 0.1 * np.abs(rng.standard_normal((n_freq, n_time)))  # positive bg
    track_bin = 20
    s[track_bin, :] += 10.0  # strong narrowband track across all time

    det = LikelihoodRatio.likelihood_ratio(s, threshold=3)

    assert det.shape == s.shape
    assert set(np.unique(det)).issubset({0, 1})

    # The track bin should be flagged across most time steps, and dominate the
    # total number of detections.
    assert det[track_bin].sum() > 0.5 * n_time
    assert det[track_bin].sum() > 0.5 * det.sum()
