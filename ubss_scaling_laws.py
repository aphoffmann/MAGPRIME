"""Monte Carlo driver for UBSS scaling law experiments.

This version keeps the simulation light-weight for validation but now
leverages the exact NSGT configuration used inside UBSS and provides first
pass scaling-law fits on the scene-level summaries.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import importlib.util
import itertools
import sys

import magpylib as magpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nsgt import CQ_NSGT
from scipy import signal
from scipy.optimize import curve_fit
from scipy.signal import chirp
import scipy.spatial.transform as st
from magprime.utility import load_swarm_data, load_michibiki_data

UBSS_MODULE_NAME = "magprime.algorithms.interference.UBSS"
UBSS_PATH = Path(__file__).resolve().parent / "magprime" / "algorithms" / "interference" / "UBSS.py"

_spec = importlib.util.spec_from_file_location(UBSS_MODULE_NAME, UBSS_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load UBSS solver from {UBSS_PATH}")
UBSS = importlib.util.module_from_spec(_spec)
sys.modules[UBSS_MODULE_NAME] = UBSS
_spec.loader.exec_module(UBSS)
calculate_delta_s = UBSS.calculate_delta_s

EPS = 1e-12
IAR_MIN = 1e-3
IAR_MAX = 1e6

MICHIBIKI_RAW = load_michibiki_data()
MICHIBIKI_DIFF = np.squeeze(MICHIBIKI_RAW[1] - MICHIBIKI_RAW[0])
MICHIBIKI_LEN = MICHIBIKI_DIFF.shape[-1]

@dataclass
class SimulationConfig:
    """Configuration block for a small-scale UBSS Monte Carlo run."""

    sample_rate: float = 50.0  # Hz
    duration_s: float = 100.0
    num_sensors: int = 3
    num_sources: int = 4
    axis: int = 0  # magnetic axis to monitor when building the mixing matrix
    seeds: Sequence[int] = field(default_factory=lambda: [0])

    ambient_amp_range: Tuple[float, float] = (25.0, 40.0)
    interference_amp_range: Tuple[float, float] = (2.0, 30.0)
    interference_dr_max: float = 100.0
    source_activity_prob: float = 0.2
    noise_std: float = 0.0

    nperseg: int = 128
    noverlap: int = 96
    magnitude_mask_threshold: float = 1e-6
    source_activity_threshold: float = 1e-8
    solver_activity_threshold: float = 1e-3
    geometry_scales: Sequence[float] = field(default_factory=lambda: [0.6])

    ubss_bpo: int = 4
    ubss_sigma: float = 5.0
    ubss_lambda: float = 1.5
    ubss_ssp_tol: float = 20.0
    ubss_cs_iters: int = 1

    output_dir: Path | None = None


@dataclass
class SceneResult:
    seed: int
    per_bin: pd.DataFrame
    scene_metrics: Dict[str, float]


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    vec = rng.normal(size=3)
    norm = np.linalg.norm(vec)
    if norm < EPS:
        return np.array([1.0, 0.0, 0.0])
    return vec / norm


def random_rotation(rng: np.random.Generator) -> st.Rotation:
    return st.Rotation.random(random_state=rng)


def make_sensors(num: int, rng: np.random.Generator, scale: float = 1.0) -> List[magpy.Sensor]:
    sensors: List[magpy.Sensor] = []
    half = 0.05 * scale
    for _ in range(num):
        face = rng.integers(0, 6)
        coords = rng.uniform(-half, half, size=3)
        axis_idx = face // 2
        coords[axis_idx] = half if face % 2 == 0 else -half
        orientation = random_rotation(rng)
        sensor = magpy.Sensor(position=coords, orientation=orientation)
        sensors.append(sensor)
    return sensors


def make_sources(num: int, rng: np.random.Generator, scale: float = 1.0) -> List[magpy.current.Loop]:
    sources: List[magpy.current.Loop] = []
    xy_span = 0.04 * scale
    z_low = 0.01 * scale
    z_high = 0.04 * scale
    for _ in range(num):
        position = np.array(
            [
                rng.uniform(-xy_span, xy_span),
                rng.uniform(-xy_span, xy_span),
                rng.uniform(z_low, z_high),
            ]
        )
        current = rng.uniform(5.0, 15.0)
        diameter = rng.uniform(0.005, 0.02) * scale
        orientation = random_rotation(rng)
        loop = magpy.current.Loop(current=current, diameter=diameter, orientation=orientation, position=position)
        sources.append(loop)
    return sources


def compute_mixing_matrix(
    sensors: Sequence[magpy.Sensor],
    sources: Sequence[magpy.misc.Dipole],
    axis: int,
) -> np.ndarray:
    columns: List[np.ndarray] = []
    for src in sources:
        field_vec = np.array(src.getB(sensors)) * 1e9  # Tesla -> nT
        columns.append(field_vec[:, axis])
    if not columns:
        return np.zeros((len(sensors), 0))
    return np.stack(columns, axis=1).astype(np.float64)


def mutual_coherence(A: np.ndarray) -> float:
    if A.size == 0:
        return 0.0
    columns = A / (np.linalg.norm(A, axis=0, keepdims=True) + EPS)
    gram = columns.T @ columns
    np.fill_diagonal(gram, 0.0)
    return float(np.max(np.abs(gram)))


def generate_ambient_signal(
    n_samples: int,
    axis: int,
    rng: np.random.Generator,
    config: SimulationConfig,
) -> np.ndarray:
    base_start = 160000
    max_offset = max(1, 2000)
    offset = int(rng.integers(0, max_offset))
    start = base_start + offset
    stop = start + n_samples
    try:
        swarm_segment = load_swarm_data(start, stop)
    except Exception:
        swarm_segment = load_swarm_data(base_start, base_start + n_samples)
    ambient = swarm_segment[axis].astype(float)
    ambient -= np.mean(ambient)
    max_abs = np.max(np.abs(ambient)) + EPS
    scale = 1.0
    ambient = ambient * (scale / max_abs)
    return ambient


def _normalize_signal(sig: np.ndarray) -> np.ndarray:
    sig = sig - np.mean(sig)
    max_abs = np.max(np.abs(sig)) + EPS
    return sig / max_abs


def _reaction_wheel_signal(fs: float, n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / fs
    base_freq = max(3, int(rng.integers(4, max(5, int(fs // 2)))))
    shift_freq = rng.uniform(1.0, base_freq)
    duration_s = rng.uniform(1.0, 5.0)
    duration_n = int(np.clip(duration_s * fs, 1, n // 2))
    shift_start = rng.integers(0, max(1, n - 2 * duration_n))
    signal_rw = np.sin(2 * np.pi * base_freq * t)
    window = np.arange(duration_n) / fs
    down = chirp(window, base_freq, duration_n / fs, shift_freq, method="hyperbolic")
    up = chirp(window, shift_freq, duration_n / fs, base_freq, method="hyperbolic")
    signal_rw[shift_start:shift_start + duration_n] = down[: min(duration_n, n - shift_start)]
    end_idx = min(shift_start + 2 * duration_n, n)
    signal_rw[shift_start + duration_n:end_idx] = up[: max(0, end_idx - (shift_start + duration_n))]
    return _normalize_signal(signal_rw)


def _michibiki_signal(n: int, axis: int, rng: np.random.Generator) -> np.ndarray:
    axis = int(np.clip(axis, 0, MICHIBIKI_DIFF.shape[0] - 1))
    data = MICHIBIKI_DIFF[axis]
    if data.shape[0] <= n:
        segment = data[:n]
    else:
        start = int(rng.integers(0, data.shape[0] - n))
        segment = data[start : start + n]
    return _normalize_signal(segment)


def _arcjet_signal(n: int, rng: np.random.Generator) -> np.ndarray:
    f = max(1, n // 300)
    switches = rng.choice([0, 1], size=(n // f + 1,), p=[0.3, 0.7])
    mask = np.ones(n)
    for idx, state in enumerate(switches):
        if state:
            mask[idx * f : min((idx + 1) * f, n)] = 0.0
    return _normalize_signal(mask)


def _sawtooth_signal(fs: float, n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / fs
    freq = rng.uniform(1.0, 5.0)
    base = signal.sawtooth(2 * np.pi * freq * t)
    mask = _arcjet_signal(n, rng)
    return _normalize_signal(base * mask)


def generate_interference_signals(
    num_sources: int,
    n_samples: int,
    fs: float,
    rng: np.random.Generator,
    amp_range: Tuple[float, float],
    dr_max: float,
    axis: int,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    base_signals = [
        _reaction_wheel_signal(fs, n_samples, rng),
        _michibiki_signal(n_samples, axis, rng),
        _arcjet_signal(n_samples, rng),
        _sawtooth_signal(fs, n_samples, rng),
    ]
    if num_sources > len(base_signals):
        repeats = int(np.ceil(num_sources / len(base_signals)))
        base_signals = (base_signals * repeats)[:num_sources]
    else:
        base_signals = base_signals[:num_sources]

    base_signals = np.array(base_signals)
    base_amplitudes = rng.uniform(*amp_range, size=len(base_signals))
    dr_targets = 10 ** rng.uniform(0.0, np.log10(dr_max + EPS), size=len(base_signals))
    amplitudes = base_amplitudes * dr_targets / max(dr_targets.max(), EPS)


    signals = amplitudes[:, None] * base_signals
    metadata: List[Dict[str, float]] = []
    for idx, base in enumerate(base_signals):
        metadata.append(
            {
                "kind": ["reaction_wheel", "michibiki", "arcjet", "sawtooth"][idx % 4],
                "amplitude": float(amplitudes[idx]),
                "activity_fraction": float(np.mean(np.abs(base) > 0.05)),
            }
        )
    return signals, metadata


def run_ubss(B: np.ndarray, config: SimulationConfig) -> Tuple[np.ndarray, Dict[str, Optional[np.ndarray]]]:
    UBSS.fs = config.sample_rate
    UBSS.bpo = config.ubss_bpo
    UBSS.sigma = config.ubss_sigma
    UBSS.lambda_ = config.ubss_lambda
    UBSS.sspTol = config.ubss_ssp_tol
    UBSS.cs_iters = config.ubss_cs_iters
    UBSS.detrend = False
    UBSS.boom = None

    UBSS.enable_analysis(True)
    estimate = UBSS.clean(np.asarray(B, dtype=float), triaxial=False)
    analysis = UBSS.get_analysis_results()
    UBSS.enable_analysis(False)

    return np.squeeze(np.asarray(estimate)), analysis


def compute_time_domain_metrics(
    ambient: np.ndarray,
    ambient_hat: np.ndarray,
    mixture: np.ndarray,
    interference: np.ndarray,
) -> Dict[str, float]:
    ambient_energy = np.sum(ambient**2)
    residual_in = mixture - ambient
    residual_out = ambient_hat - ambient

    snr_in = 10.0 * np.log10((ambient_energy + EPS) / (np.sum(residual_in**2) + EPS))
    snr_out = 10.0 * np.log10((ambient_energy + EPS) / (np.sum(residual_out**2) + EPS))
    delta_snr = snr_out - snr_in
    nrmse = np.linalg.norm(residual_out) / (np.linalg.norm(ambient) + EPS)

    interference_energy = np.sum(interference**2)
    iar_scene = interference_energy / (ambient_energy + EPS)

    return {
        "SNR_in": float(snr_in),
        "SNR_out": float(snr_out),
        "Delta_SNR": float(delta_snr),
        "nRMSE": float(nrmse),
        "IAR_scene": float(iar_scene),
    }


def nsgt_forward_stack(
    signals: np.ndarray,
    config: SimulationConfig,
) -> Tuple[np.ndarray, np.ndarray, CQ_NSGT]:
    array = np.asarray(signals)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    length = array.shape[-1]
    bpo = config.ubss_bpo
    fs = config.sample_rate
    fmax = fs / 2.0
    lowf = 2.0 * bpo * fs / length
    nsgt_obj = CQ_NSGT(lowf, fmax, bpo, fs, length, multichannel=True)

    coeffs = nsgt_obj.forward(array)
    coeffs = np.array(coeffs, dtype=object)

    stacked = np.vstack([np.hstack(coeffs[i]) for i in range(coeffs.shape[0])])
    subband_lengths = np.array([band.shape[-1] for band in coeffs[0]])

    band_indices = np.concatenate(
        [np.full(length, idx, dtype=int) for idx, length in enumerate(subband_lengths)]
    )
    weight_vec = 1.0 / np.sqrt(subband_lengths[band_indices].astype(float) + EPS)
    stacked = stacked * weight_vec

    time_energy = np.sum(np.abs(array) ** 2)
    tf_energy = np.sum(np.abs(stacked) ** 2)
    if tf_energy > 0:
        scale = np.sqrt((time_energy + EPS) / (tf_energy + EPS))
        stacked = stacked * scale

    return stacked, subband_lengths, nsgt_obj


def flatten_tf(matrix: np.ndarray) -> np.ndarray:
    return matrix.reshape(matrix.shape[0], -1)


def wrap_phase(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def plot_simulation_stft(
    ambient: np.ndarray,
    mixture: np.ndarray,
    interference: np.ndarray,
    config: SimulationConfig,
    seed: int,
) -> None:
    """Display STFT magnitude snapshots for the simulated signals."""
    ambient_1d = np.asarray(ambient).reshape(-1)
    mixture_array = np.asarray(mixture)
    if mixture_array.ndim == 1:
        mixture_array = mixture_array[np.newaxis, :]
    interference_array = np.asarray(interference)
    if interference_array.ndim == 1:
        interference_array = interference_array[np.newaxis, :]

    entries = [
        ("Ambient", ambient_1d),
        ("Mixture sensor 0", mixture_array[0]),
        ("Interference sensor 0", interference_array[0]),
    ]

    fig, axes = plt.subplots(len(entries), 1, figsize=(10, 3.0 * len(entries)), sharex=True)
    if len(entries) == 1:
        axes = [axes]

    for ax, (label, data) in zip(axes, entries):
        freqs, times, Zxx = signal.stft(
            data,
            fs=config.sample_rate,
            nperseg=config.nperseg,
            noverlap=config.noverlap,
        )
        magnitude_db = 20.0 * np.log10(np.abs(Zxx) + EPS)
        mesh = ax.pcolormesh(times, freqs, magnitude_db, shading="auto")
        ax.set_ylabel("Hz")
        ax.set_title(f"{label} STFT (seed {seed})")
        fig.colorbar(mesh, ax=ax, label="Magnitude (dB)")

    axes[-1].set_xlabel("Seconds")
    fig.tight_layout()
    plt.show()



def make_scaling_plots(
    per_bin: pd.DataFrame,
    fits: Dict[str, Dict[str, float]] | None = None,
    output_path: Path | None = None,
    show: bool = True,
) -> None:
    """Render the canonical scaling-law scatter plots with optional model overlays."""
    if per_bin.empty or "error_ratio_bin" not in per_bin:
        return

    error_ratio = per_bin["error_ratio_bin"].to_numpy(dtype=float)
    error_db = 10.0 * np.log10(error_ratio + EPS)

    rho_series = per_bin.get("rho_k", pd.Series(dtype=float))
    rho = rho_series.to_numpy(dtype=float)
    mu_support_series = per_bin.get("mu_support", pd.Series(dtype=float))
    mu_series = per_bin.get("mu_A", pd.Series(dtype=float))
    mu_support_vals = mu_support_series.to_numpy(dtype=float)
    mu_vals = np.where(mu_support_vals > 0, mu_support_vals, mu_series.to_numpy(dtype=float))

    dr_proxy_series = per_bin.get("DR_proxy_bin", pd.Series(dtype=float))
    dr_proxy = dr_proxy_series.to_numpy(dtype=float)
    dr_db = 20.0 * np.log10(np.maximum(dr_proxy, EPS))

    iar_series = per_bin.get("IAR_bin", pd.Series(dtype=float))
    iar = iar_series.to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_rho, ax_mu, ax_dr, ax_iar = axes.flatten()

    scatter_kwargs = dict(s=12, alpha=0.35, edgecolor="none")

    ax_rho.scatter(rho, error_db, **scatter_kwargs)
    ax_rho.set_xlabel(r"$\rho_k$")
    ax_rho.set_ylabel("Residual energy (dB rel. ambient)")
    ax_rho.axvline(1.0, color="k", linestyle="--", linewidth=1, label="k = M")

    ax_mu.scatter(mu_vals, error_db, **scatter_kwargs)
    ax_mu.set_xlabel(r"$\mu(A)$")
    ax_mu.set_ylabel("Residual energy (dB rel. ambient)")

    ax_dr.scatter(dr_db, error_db, **scatter_kwargs)
    ax_dr.set_xlabel("DR proxy (dB)")
    ax_dr.set_ylabel("Residual energy (dB rel. ambient)")

    ax_iar.scatter(iar, error_db, **scatter_kwargs)
    ax_iar.set_xlabel("IAR")
    ax_iar.set_ylabel("Residual energy (dB rel. ambient)")
    ax_iar.set_xscale("log")

    def mu_transform(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 1e-6, 1.0 - 1e-6)
        return np.log(1.0 / (1.0 - x + EPS))

    if fits:
        log_fit = fits.get("log_model_overall", {})
        if log_fit.get("status") == "ok":
            def predict_error_db(iar_val, dr_val, mu_val, rho_val):
                mu_t = mu_transform(mu_val)
                log_err = (
                    log_fit["c0"]
                    + log_fit["alpha"] * np.log(np.clip(iar_val, 1e-6, 1e6))
                    + log_fit["beta"] * np.log(np.maximum(dr_val, EPS))
                    + log_fit["gamma"] * mu_t
                    + log_fit["eta"] * np.log(np.maximum(rho_val, EPS))
                )
                return (10.0 / np.log(10.0)) * log_err

            med_iar = float(np.nanmedian(iar[iar > 0])) if np.any(iar > 0) else 1.0
            med_dr = float(np.nanmedian(dr_proxy[dr_proxy > 0])) if np.any(dr_proxy > 0) else 1.0
            med_mu = float(np.nanmedian(mu_vals[(mu_vals > 0) & (mu_vals < 1.0)])) if np.any((mu_vals > 0) & (mu_vals < 1.0)) else 0.5
            med_rho = float(np.nanmedian(rho[rho > 0])) if np.any(rho > 0) else 0.5

            rho_positive = rho[rho > 0]
            rho_min = float(np.nanmin(rho_positive)) if rho_positive.size else EPS
            rho_max = float(np.nanmax(rho_positive)) if rho_positive.size else 2.0
            rho_grid = np.linspace(max(rho_min, EPS), max(rho_min, min(rho_max, 2.0)), 200)
            ax_rho.plot(rho_grid, predict_error_db(med_iar, med_dr, med_mu, rho_grid), color="tab:red", linewidth=2, label="log-fit")

            mu_positive = mu_vals[(mu_vals > 0) & (mu_vals < 1.0)]
            mu_min = float(np.nanmin(mu_positive)) if mu_positive.size else 1e-3
            mu_max = float(np.nanmax(mu_positive)) if mu_positive.size else 0.999
            mu_grid = np.linspace(max(mu_min, 1e-3), min(mu_max, 0.999), 200)
            ax_mu.plot(mu_grid, predict_error_db(med_iar, med_dr, mu_grid, med_rho), color="tab:red", linewidth=2, label="log-fit")

            dr_positive = dr_proxy[dr_proxy > 0]
            dr_min = float(dr_positive.min()) if dr_positive.size else 1.0
            dr_max = float(dr_proxy.max()) if dr_proxy.size else dr_min * 10
            if dr_max <= dr_min:
                dr_max = dr_min * 1.1
            dr_grid = np.logspace(np.log10(max(dr_min, 1.0)), np.log10(max(dr_max, dr_min * 1.01)), 200)
            ax_dr.plot(20.0 * np.log10(dr_grid), predict_error_db(med_iar, dr_grid, med_mu, med_rho), color="tab:red", linewidth=2, label="log-fit")

            iar_positive = iar[iar > 0]
            iar_min = float(iar_positive.min()) if iar_positive.size else 1e-6
            iar_max = float(iar.max()) if iar.size else iar_min * 10
            if iar_max <= iar_min:
                iar_max = iar_min * 10
            iar_grid = np.logspace(np.log10(max(iar_min, 1e-6)), np.log10(iar_max), 200)
            ax_iar.plot(iar_grid, predict_error_db(iar_grid, med_dr, med_mu, med_rho), color="tab:red", linewidth=2, label="log-fit")

        sat_fit = fits.get("saturating_model", {})
        if sat_fit.get("status") == "ok":
            med_mu = float(np.nanmedian(mu_vals[(mu_vals > 0) & (mu_vals < 1.0)])) if np.any((mu_vals > 0) & (mu_vals < 1.0)) else 0.5
            med_rho = float(np.nanmedian(rho[rho > 0])) if np.any(rho > 0) else 0.5
            iar_positive = iar[iar > 0]
            iar_min = float(iar_positive.min()) if iar_positive.size else 1e-6
            iar_max = float(iar.max()) if iar.size else iar_min * 10
            if iar_max <= iar_min:
                iar_max = iar_min * 10
            iar_grid = np.logspace(np.log10(max(iar_min, 1e-6)), np.log10(iar_max), 200)
            mu_array = np.full_like(iar_grid, med_mu)
            rho_array = np.full_like(iar_grid, med_rho)
            params = (
                float(sat_fit["log_e_min"]),
                float(sat_fit["log_e_span"]),
                float(sat_fit["I0"]),
                float(sat_fit["p"]),
                float(sat_fit["gamma"]),
                float(sat_fit["eta"]),
            )
            log_error_pred = _log_error_model((iar_grid, mu_array, rho_array), *params)
            error_db_pred = (10.0 / np.log(10.0)) * log_error_pred
            ax_iar.plot(iar_grid, error_db_pred, color="tab:green", linewidth=2, linestyle="--", label="saturating-fit")

    for axis in (ax_rho, ax_mu, ax_dr, ax_iar):
        axis.grid(True, linestyle=":", alpha=0.4)
        if len(axis.get_lines()) > 0:
            axis.legend(loc="best")

    fig.suptitle("UBSS Scaling Law Diagnostics", fontsize=16)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)


def build_per_bin_table(
    seed: int,
    ambient_tf: np.ndarray,
    ambient_hat_tf: np.ndarray,
    mixture_tf: np.ndarray,
    interference_tf: np.ndarray,
    source_tf: np.ndarray,
    solver_info: Dict[str, Optional[np.ndarray]],
    mixing_matrix: np.ndarray,
    geometry_scale: float,
    config: SimulationConfig,
) -> pd.DataFrame:
    ambient_flat = ambient_tf.flatten()
    ambient_hat_flat = ambient_hat_tf.flatten()
    mixture_flat = flatten_tf(mixture_tf)
    interference_flat = flatten_tf(interference_tf)
    source_flat = flatten_tf(source_tf)

    n_bins = ambient_flat.size
    ambient_energy = np.abs(ambient_flat) ** 2
    error_energy = np.abs(ambient_flat - ambient_hat_flat) ** 2
    snr_bins = 10.0 * np.log10((ambient_energy + EPS) / (error_energy + EPS))
    mag_re = np.abs(np.abs(ambient_hat_flat) - np.abs(ambient_flat)) / (np.abs(ambient_flat) + EPS)
    phase_err = wrap_phase(np.angle(ambient_hat_flat) - np.angle(ambient_flat))

    interference_energy = np.sum(np.abs(interference_flat) ** 2, axis=0)
    iar_bins = interference_energy / (ambient_energy + EPS)
    iar_bins = np.clip(iar_bins, IAR_MIN, IAR_MAX)

    snr_in_bins = 10.0 * np.log10((ambient_energy + EPS) / (interference_energy + EPS))
    delta_snr_bins = snr_bins - snr_in_bins

    ambient_positive = ambient_energy[ambient_energy > 0]
    ambient_floor = float(np.percentile(ambient_positive, 20)) if ambient_positive.size else 0.0
    ambient_threshold = max(ambient_floor, 1e-10)
    valid_mask = (ambient_energy > ambient_threshold) & np.isfinite(iar_bins)
    valid_mask &= np.isfinite(snr_bins)

    source_magnitudes = np.abs(source_flat)
    if source_magnitudes.size:
        oracle_active = np.sum(source_magnitudes > config.source_activity_threshold, axis=0)
        largest = np.max(source_magnitudes, axis=0)
        if source_flat.shape[0] >= 2:
            sorted_vals = np.sort(source_magnitudes, axis=0)
            second_largest = sorted_vals[-2]
        else:
            second_largest = np.zeros_like(largest)
        dr_proxy = np.where(
            second_largest > config.source_activity_threshold,
            largest / (second_largest + EPS),
            1.0,
        )
    else:
        oracle_active = np.zeros(n_bins)
        dr_proxy = np.ones(n_bins)

    mu_global = mutual_coherence(mixing_matrix)

    solver_coeffs = None
    solver_mixing = None
    if solver_info:
        solver_coeffs = solver_info.get("coefficients")
        solver_mixing = solver_info.get("mixing_matrix")
        if solver_coeffs is not None:
            solver_coeffs = np.asarray(solver_coeffs)
        if solver_mixing is not None:
            solver_mixing = np.asarray(solver_mixing)

    k_active = np.zeros(n_bins, dtype=int)
    rho_k = np.zeros(n_bins)
    mu_support = np.full(n_bins, np.nan, dtype=float)
    cond_vals = np.full(n_bins, np.nan, dtype=float)
    delta_vals = np.full(n_bins, np.nan, dtype=float)
    cluster_labels = np.full(n_bins, -1, dtype=int)
    support_indices_list: List[Tuple[int, ...]] = [tuple() for _ in range(n_bins)]

    solver_threshold = getattr(config, "solver_activity_threshold", config.source_activity_threshold)
    if solver_coeffs is not None and solver_mixing is not None:
        num_sensors = solver_mixing.shape[0]
        for idx in range(n_bins):
            coeff_vec = solver_coeffs[:, idx]
            magnitudes = np.abs(coeff_vec)
            support_idx = np.flatnonzero(magnitudes > solver_threshold)
            A_sup = solver_mixing[:, support_idx] if support_idx.size else None
            if support_idx.size:
                dominant = support_idx[np.argmax(magnitudes[support_idx])]
                cluster_labels[idx] = int(dominant)
            support_indices_list[idx] = tuple(int(i) for i in support_idx.tolist())
            interferer_support = support_idx[support_idx != 0]
            k_active[idx] = interferer_support.size
            rho_k[idx] = interferer_support.size / max(num_sensors, 1)

            if support_idx.size >= 2:
                A_sup = solver_mixing[:, support_idx]
                norms = np.linalg.norm(A_sup, axis=0, keepdims=True) + EPS
                normalized = A_sup / norms
                gram = normalized.conj().T @ normalized
                np.fill_diagonal(gram, 0.0)
                mu_support[idx] = float(np.max(np.abs(gram)))
                cond_vals[idx] = float(
                    np.linalg.cond(A_sup.conj().T @ A_sup + EPS * np.eye(A_sup.shape[1]))
                )
            elif support_idx.size == 1:
                mu_support[idx] = np.nan
                cond_vals[idx] = 1.0
            else:
                cond_vals[idx] = 1.0

            if support_idx.size:
                delta_vals[idx] = float(calculate_delta_s(solver_mixing, coeff_vec))
    else:
        k_active = oracle_active.astype(int)
        rho_k = k_active / max(mixing_matrix.shape[0], 1)
        mu_support[:] = np.nan
        support_indices_list = [tuple() for _ in range(n_bins)]
        cond_vals[:] = (
            float(np.linalg.cond(mixing_matrix.T @ mixing_matrix + EPS * np.eye(mixing_matrix.shape[1])))
            if mixing_matrix.size
            else 1.0
        )
        delta_vals[:] = np.nan

    ssp_mask = k_active <= 1
    mag_mask = np.linalg.norm(mixture_flat, axis=0) > config.magnitude_mask_threshold

    records: List[Dict[str, object]] = []
    valid_indices = np.nonzero(valid_mask)[0]
    for idx in valid_indices:
        mu_value = mu_support[idx]
        if not np.isfinite(mu_value):
            mu_value = mu_global
        cond_value = cond_vals[idx] if np.isfinite(cond_vals[idx]) else 1.0
        delta_value = delta_vals[idx] if np.isfinite(delta_vals[idx]) else np.nan
        records.append(
            {
                "seed": seed,
                "bin_index": len(records),
                "original_bin_index": int(idx),
                "ambient_truth": ambient_flat[idx],
                "ambient_estimate": ambient_hat_flat[idx],
                "mixture_vector": mixture_flat[:, idx],
                "magnitude_mask": bool(mag_mask[idx]),
                "ssp_mask": bool(ssp_mask[idx]),
                "cluster_label": int(cluster_labels[idx]),
                "support_indices": support_indices_list[idx],
                "A_id": 0,
                "mu_A": mu_global,
                "mu_support": float(mu_value),
                "cond_AHA": float(cond_value),
                "delta_s": float(delta_value) if np.isfinite(delta_value) else np.nan,
                "SNR_bin": snr_bins[idx],
                "SNR_in_bin": snr_in_bins[idx],
                "Delta_SNR_bin": delta_snr_bins[idx],
                "error_energy_bin": float(error_energy[idx]),
                "error_ratio_bin": float(error_energy[idx] / (ambient_energy[idx] + EPS)),
                "MagRE_bin": mag_re[idx],
                "phase_error": phase_err[idx],
                "IAR_bin": iar_bins[idx],
                "DR_proxy_bin": dr_proxy[idx],
                "k_active": int(k_active[idx]),
                "rho_k": rho_k[idx],
                "geometry_scale": geometry_scale,
                "energy_weight": float(ambient_energy[idx]),
            }
        )

    return pd.DataFrame.from_records(records)


def simulate_seed(seed: int, config: SimulationConfig, show_stft: bool = False, geometry_scale: float = 1.0) -> SceneResult:
    rng = np.random.default_rng(seed)
    n_samples = int(config.sample_rate * config.duration_s)

    sensors = make_sensors(config.num_sensors, rng, scale=geometry_scale)
    sources = make_sources(config.num_sources, rng, scale=geometry_scale)
    mixing_matrix = compute_mixing_matrix(sensors, sources, config.axis)

    ambient = generate_ambient_signal(n_samples, config.axis, rng, config)
    ambient_per_sensor = np.outer(np.ones(config.num_sensors), ambient)

    source_signals, metadata = generate_interference_signals(
        config.num_sources,
        n_samples,
        config.sample_rate,
        rng,
        config.interference_amp_range,
        config.interference_dr_max,
        config.axis,
    )

    interference = mixing_matrix @ source_signals
    mixture = ambient_per_sensor + interference
    if config.noise_std > 0:
        mixture += rng.normal(scale=config.noise_std, size=mixture.shape)

    if show_stft:
        plot_simulation_stft(ambient, mixture, interference, config, seed)

    ambient_hat, solver_info = run_ubss(mixture, config)
    if ambient_hat.ndim == 2:
        ambient_hat = ambient_hat[0]

    metrics = compute_time_domain_metrics(ambient, ambient_hat, mixture[0], interference[0])

    amplitudes = np.array([item["amplitude"] for item in metadata]) if metadata else np.array([])
    pos_amplitudes = amplitudes[amplitudes > EPS]
    dr_scene = float(pos_amplitudes.max() / (pos_amplitudes.min() + EPS)) if pos_amplitudes.size else 1.0
    metrics["DR_scene"] = dr_scene

    ambient_tf, _, _ = nsgt_forward_stack(ambient, config)
    ambient_hat_tf, _, _ = nsgt_forward_stack(ambient_hat, config)
    mixture_tf, _, _ = nsgt_forward_stack(mixture, config)
    interference_tf, _, _ = nsgt_forward_stack(interference, config)
    source_tf, _, _ = nsgt_forward_stack(source_signals, config)

    per_bin = build_per_bin_table(
        seed,
        ambient_tf,
        ambient_hat_tf,
        mixture_tf,
        interference_tf,
        source_tf,
        solver_info,
        mixing_matrix,
        geometry_scale,
        config,
    )

    weights_full = np.abs(ambient_tf.flatten()) ** 2
    if per_bin.empty:
        weights_valid = np.array([], dtype=float)
    else:
        bin_indices = per_bin["original_bin_index"].to_numpy(dtype=int)
        weights_valid = weights_full[bin_indices]
        per_bin["energy_weight"] = weights_valid

    if weights_valid.size > 0 and np.sum(weights_valid) > 0:
        metrics["SNR_out_energy_weighted"] = float(np.sum(weights_valid * per_bin["SNR_bin"]) / np.sum(weights_valid))
    else:
        metrics["SNR_out_energy_weighted"] = metrics["SNR_out"]

    support_mask = per_bin["k_active"] >= 1
    if support_mask.any():
        mu_support_median = float(per_bin.loc[support_mask, "mu_support"].median())
        mu_support_p90 = float(per_bin.loc[support_mask, "mu_support"].quantile(0.9))
    else:
        mu_support_median = 0.0
        mu_support_p90 = 0.0

    metrics.update(
        {
            "seed": seed,
            "mu_A": mutual_coherence(mixing_matrix),
            "mu_support_median": mu_support_median,
            "mu_support_p90": mu_support_p90,
            "cond_AHA": float(np.linalg.cond(mixing_matrix.T @ mixing_matrix + EPS * np.eye(config.num_sources)))
            if mixing_matrix.size
            else 1.0,
            "rho_k_mean": float(per_bin["rho_k"].mean()),
            "rho_k_median": float(per_bin["rho_k"].median()),
            "rho_k_p90": float(per_bin["rho_k"].quantile(0.9)),
            "phase_error_median": float(per_bin["phase_error"].median()),
            "phase_error_p90": float(per_bin["phase_error"].quantile(0.9)),
            "ssp_fraction": float(per_bin["ssp_mask"].mean()),
            "geometry_scale": geometry_scale,
        }
    )

    return SceneResult(seed=seed, per_bin=per_bin, scene_metrics=metrics)




def fit_log_log_model(
    per_bin: pd.DataFrame,
    mask: Optional[pd.Series] = None,
    eps: float = 1e-9,
    label: str = "overall",
) -> Dict[str, float]:
    if per_bin.empty:
        return {"status": "insufficient_data", "samples": 0, "label": label}

    required = [
        "error_ratio_bin",
        "IAR_bin",
        "DR_proxy_bin",
        "mu_support",
        "mu_A",
        "rho_k",
        "energy_weight",
    ]
    missing = [col for col in required if col not in per_bin.columns]
    if missing:
        return {"status": "missing_columns", "missing": missing, "label": label}

    df = per_bin[required].replace([np.inf, -np.inf], np.nan)
    if mask is not None:
        mask = mask.reindex(per_bin.index, fill_value=False)
        df = df[mask]
    df = df.dropna()
    if df.empty:
        return {"status": "insufficient_data", "samples": 0, "label": label}

    weights = df["energy_weight"].to_numpy(dtype=float)
    positive = weights > 0
    df = df[positive]
    weights = weights[positive]
    if weights.size == 0:
        return {"status": "insufficient_data", "samples": 0, "label": label}

    threshold = np.quantile(weights, 0.2)
    keep = weights > threshold
    df = df[keep]
    weights = weights[keep]
    if len(df) < 5:
        return {"status": "insufficient_data", "samples": int(len(df)), "label": label}

    error = df["error_ratio_bin"].to_numpy(dtype=float)
    positive_error = error > 0
    df = df[positive_error]
    weights = weights[positive_error]
    if len(df) < 5:
        return {"status": "insufficient_data", "samples": int(len(df)), "label": label}

    mu_values = np.where(df["mu_support"].to_numpy() > 0, df["mu_support"], df["mu_A"])
    mu_values = np.clip(mu_values, eps, 1.0 - eps)
    mu_trans = np.log(1.0 / (1.0 - mu_values + eps))

    iar = np.clip(df["IAR_bin"].to_numpy(dtype=float), 1e-6, 1e6)
    dr = np.clip(df["DR_proxy_bin"].to_numpy(dtype=float), eps, None)
    rho = np.clip(df["rho_k"].to_numpy(dtype=float), eps, None)

    y = np.log(df["error_ratio_bin"].to_numpy(dtype=float) + eps)
    X = np.column_stack(
        [
            np.ones(len(df)),
            np.log(iar),
            np.log(dr),
            mu_trans,
            np.log(rho),
        ]
    )

    w = weights / np.mean(weights)
    sqrt_w = np.sqrt(w)
    X_weighted = X * sqrt_w[:, None]
    y_weighted = y * sqrt_w

    coeffs, residuals, rank, _ = np.linalg.lstsq(X_weighted, y_weighted, rcond=None)
    preds = X @ coeffs
    residual_vec = y - preds
    dof = max(len(df) - X.shape[1], 1)
    if residuals.size > 0:
        sigma2 = residuals[0] / dof
    else:
        sigma2 = float((residual_vec * w) @ residual_vec / dof)

    XtX = X.T @ (w[:, None] * X)
    XtX_inv = np.linalg.inv(XtX)
    se = np.sqrt(np.diag(XtX_inv * sigma2))

    y_mean = np.average(y, weights=w)
    ss_tot = np.sum(w * (y - y_mean) ** 2)
    ss_res = np.sum(w * residual_vec**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {
        "status": "ok",
        "samples": int(len(df)),
        "label": label,
        "mu_feature": "mu_support_fallback",
        "target": "Delta_SNR_bin",
        "c0": float(coeffs[0]),
        "alpha": float(coeffs[1]),
        "beta": float(coeffs[2]),
        "gamma": float(coeffs[3]),
        "eta": float(coeffs[4]),
        "c0_se": float(se[0]),
        "alpha_se": float(se[1]),
        "beta_se": float(se[2]),
        "gamma_se": float(se[3]),
        "eta_se": float(se[4]),
        "r2": float(r2),
    }


def _log_error_model(x, log_e_min, log_e_span, I0, p, gamma, eta):
    IAR, mu, rho = x
    base = log_e_min + log_e_span / (1.0 + (IAR / (I0 + EPS)) ** p)
    penalty = gamma * np.log(mu + EPS) + eta * np.log(rho + EPS)
    return base + penalty




def fit_saturating_model(per_bin: pd.DataFrame, eps: float = 1e-9) -> Dict[str, float]:
    if per_bin.empty:
        return {"status": "insufficient_data", "samples": 0}

    required = ["error_ratio_bin", "IAR_bin", "mu_support", "mu_A", "rho_k", "energy_weight"]
    missing = [col for col in required if col not in per_bin.columns]
    if missing:
        return {"status": "missing_columns", "missing": missing}

    df = per_bin[required].replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        return {"status": "insufficient_data", "samples": 0}

    weights = df["energy_weight"].to_numpy(dtype=float)
    positive = weights > 0
    df = df[positive]
    weights = weights[positive]
    if weights.size == 0:
        return {"status": "insufficient_data", "samples": 0}

    threshold = np.quantile(weights, 0.2)
    keep = weights > threshold
    df = df[keep]
    weights = weights[keep]
    if len(df) < 6:
        return {"status": "insufficient_data", "samples": int(len(df))}

    mu_values = np.where(df["mu_support"].to_numpy() > 0, df["mu_support"], df["mu_A"])
    mu_mask = mu_values > 0
    df = df[mu_mask]
    weights = weights[mu_mask]
    mu_values = mu_values[mu_mask]
    if mu_values.size < 6:
        return {"status": "insufficient_data", "samples": int(mu_values.size)}

    IAR = np.clip(df["IAR_bin"].to_numpy(dtype=float), 1e-6, 1e6)
    rho = np.clip(df["rho_k"].to_numpy(dtype=float), 1e-6, None)
    log_error = np.log(df["error_ratio_bin"].to_numpy(dtype=float) + eps)
    mu_values = np.clip(mu_values, 1e-6, 1.0 - 1e-6)

    finite_mask = np.isfinite(IAR) & np.isfinite(mu_values) & np.isfinite(rho) & np.isfinite(log_error)
    IAR = IAR[finite_mask]
    mu_values = mu_values[finite_mask]
    rho = rho[finite_mask]
    log_error = log_error[finite_mask]
    weights = weights[finite_mask]

    if log_error.size < 6:
        return {"status": "insufficient_data", "samples": int(log_error.size), "mu_feature": "mu_support_fallback"}

    positive_IAR = IAR[IAR > 0]
    I0_guess = np.median(positive_IAR) if positive_IAR.size else 1.0
    log_e_min = float(np.percentile(log_error, 10))
    log_e_max = float(np.percentile(log_error, 90))
    log_e_span = max(log_e_max - log_e_min, 1e-4)

    initial = [log_e_min, log_e_span, float(np.clip(I0_guess, 1e-6, 1e6)), 1.0, 0.0, 0.0]
    bounds = (
        [-20.0, 1e-6, 1e-6, 0.5, -20.0, -20.0],
        [20.0, 20.0, 1e6, 3.0, 20.0, 20.0],
    )

    w = weights / np.mean(weights)
    sigma = 1.0 / np.sqrt(np.maximum(w, 1e-12))

    try:
        popt, pcov = curve_fit(
            _log_error_model,
            (IAR, mu_values, rho),
            log_error,
            p0=initial,
            bounds=bounds,
            sigma=sigma,
            absolute_sigma=True,
            maxfev=40000,
        )
        perr = np.sqrt(np.diag(pcov))
        return {
            "status": "ok",
            "samples": int(log_error.size),
            "mu_feature": "mu_support_fallback",
            "target": "Delta_SNR_bin",
            "log_e_min": float(popt[0]),
            "log_e_span": float(popt[1]),
            "I0": float(popt[2]),
            "p": float(popt[3]),
            "gamma": float(popt[4]),
            "eta": float(popt[5]),
            "log_e_min_se": float(perr[0]),
            "log_e_span_se": float(perr[1]),
            "I0_se": float(perr[2]),
            "p_se": float(perr[3]),
            "gamma_se": float(perr[4]),
            "eta_se": float(perr[5]),
        }
    except Exception as exc:  # pragma: no cover - debug aid
        return {
            "status": "fit_failed",
            "samples": int(log_error.size),
            "mu_feature": "mu_support_fallback",
            "message": str(exc),
        }


def fit_scaling_models(per_bin: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    mask_sparse = per_bin.get("rho_k") <= 1.0 if "rho_k" in per_bin else None
    mask_dense = per_bin.get("rho_k") > 1.0 if "rho_k" in per_bin else None

    results["log_model_overall"] = fit_log_log_model(per_bin, label="overall")
    if mask_sparse is not None:
        results["log_model_sparse"] = fit_log_log_model(per_bin, mask=mask_sparse, label="rho_leq_1")
    if mask_dense is not None:
        results["log_model_dense"] = fit_log_log_model(per_bin, mask=mask_dense, label="rho_gt_1")
    results["saturating_model"] = fit_saturating_model(per_bin)
    return results


def run_simulation(
    config: SimulationConfig, show_stft: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]]]:
    scene_results: List[SceneResult] = []
    combinations = list(itertools.product(enumerate(config.geometry_scales), enumerate(config.seeds)))
    for combo_idx, ((geo_idx, scale), (seed_idx, seed)) in enumerate(combinations):
        combined_seed = seed + int(1_000 * geo_idx)
        scene_results.append(
            simulate_seed(
                combined_seed,
                config,
                show_stft=bool(show_stft and combo_idx == 0),
                geometry_scale=scale,
            )
        )

    per_bin_tables = [scene.per_bin for scene in scene_results]
    per_scene_rows = [scene.scene_metrics for scene in scene_results]

    per_bin = pd.concat(per_bin_tables, ignore_index=True)
    per_scene = pd.DataFrame(per_scene_rows)

    if config.output_dir is not None:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        per_bin_file = config.output_dir / "per_bin_combined.parquet"
        scene_file = config.output_dir / "scene_metrics.csv"
        per_bin.to_parquet(per_bin_file, index=False)
        per_scene.to_csv(scene_file, index=False)

    fit_results = fit_scaling_models(per_bin)

    return per_bin, per_scene, fit_results


def run_small_demo(
    show_stft: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]]]:
    config = SimulationConfig()
    return run_simulation(config, show_stft=show_stft)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="UBSS scaling law demo")
    parser.add_argument("--show-stft", action="store_true", help="Display STFT plots for the first seed")
    parser.add_argument("--plot-scaling", action="store_true", help="Show the four canonical scaling-law plots")
    parser.add_argument("--plot-dir", type=Path, help="Directory to save scaling-law figures")
    args = parser.parse_args()

    per_bin_df, scene_df, fits = run_small_demo(show_stft=args.show_stft)
    print("Per-scene summary:\n", scene_df.head())
    print("Per-bin preview:\n", per_bin_df.head())
    print("Log-log fit (overall):", fits.get("log_model_overall"))
    if "log_model_sparse" in fits:
        print("  Sparse regime:", fits["log_model_sparse"])
    if "log_model_dense" in fits:
        print("  Dense regime:", fits["log_model_dense"])
    print("Saturating fit:", fits["saturating_model"])

    plot_path = None
    if args.plot_dir is not None:
        plot_path = args.plot_dir / "ubss_scaling_quad.png"

    if args.plot_scaling or plot_path is not None:
        make_scaling_plots(per_bin_df, fits=fits, output_path=plot_path, show=args.plot_scaling)
