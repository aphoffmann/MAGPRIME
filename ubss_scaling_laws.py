"""Monte Carlo driver for UBSS scaling law experiments.

This version keeps the simulation light-weight for validation but now
leverages the exact NSGT configuration used inside UBSS and provides first
pass scaling-law fits on the scene-level summaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import magpylib as magpy
import numpy as np
import pandas as pd
from nsgt import CQ_NSGT
from scipy import signal
from scipy.optimize import curve_fit
from scipy.signal import chirp
import scipy.spatial.transform as st

from magprime.algorithms.interference import UBSS
from magprime.algorithms.interference.UBSS import calculate_delta_s

EPS = 1e-12


@dataclass
class SimulationConfig:
    """Configuration block for a small-scale UBSS Monte Carlo run."""

    sample_rate: float = 256.0  # Hz
    duration_s: float = 4.0
    num_sensors: int = 3
    num_sources: int = 3
    axis: int = 0  # magnetic axis to monitor when building the mixing matrix
    seeds: Sequence[int] = field(default_factory=lambda: [0])

    ambient_amp_range: Tuple[float, float] = (25.0, 40.0)
    interference_amp_range: Tuple[float, float] = (2.0, 30.0)
    interference_dr_max: float = 100.0
    source_activity_prob: float = 0.35
    noise_std: float = 0.0

    nperseg: int = 128
    noverlap: int = 96
    magnitude_mask_threshold: float = 1e-6
    source_activity_threshold: float = 1e-8

    ubss_bpo: int = 16
    ubss_sigma: float = 5.0
    ubss_lambda: float = 1.5
    ubss_ssp_tol: float = 20.0
    ubss_cs_iters: int = 3

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


def make_sensors(num: int, rng: np.random.Generator) -> List[magpy.Sensor]:
    sensors: List[magpy.Sensor] = []
    for _ in range(num):
        position = rng.uniform(low=[-0.4, -0.4, 0.05], high=[0.4, 0.4, 0.55])
        orientation = random_rotation(rng)
        sensor = magpy.Sensor(position=position, orientation=orientation)
        sensors.append(sensor)
    return sensors


def make_sources(num: int, rng: np.random.Generator) -> List[magpy.misc.Dipole]:
    sources: List[magpy.misc.Dipole] = []
    for _ in range(num):
        position = rng.uniform(low=[-0.5, -0.5, -0.2], high=[0.5, 0.5, 0.3])
        moment_direction = random_unit_vector(rng)
        dipole = magpy.misc.Dipole(moment=moment_direction, position=position)
        sources.append(dipole)
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
    n_samples: int, fs: float, rng: np.random.Generator, amp_range: Tuple[float, float]
) -> np.ndarray:
    t = np.arange(n_samples) / fs
    num_components = 3
    signal_sum = np.zeros_like(t)
    for _ in range(num_components):
        freq = rng.uniform(0.05, 2.5)
        amp = rng.uniform(*amp_range)
        phase = rng.uniform(0, 2 * np.pi)
        signal_sum += amp * np.sin(2 * np.pi * freq * t + phase)
    trend = rng.uniform(-0.5, 0.5) * t
    return signal_sum + trend


def make_activity_mask(n_samples: int, block: int, rng: np.random.Generator, p: float) -> np.ndarray:
    num_blocks = int(np.ceil(n_samples / block))
    active_blocks = rng.random(num_blocks) < p
    mask = np.repeat(active_blocks, block)
    return mask[:n_samples]


def generate_interference_signals(
    num_sources: int,
    n_samples: int,
    fs: float,
    rng: np.random.Generator,
    amp_range: Tuple[float, float],
    dr_max: float,
    activity_prob: float,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    t = np.arange(n_samples) / fs
    block = max(int(0.05 * fs), 1)
    signals: List[np.ndarray] = []
    metadata: List[Dict[str, float]] = []

    base_amplitudes = rng.uniform(*amp_range, size=num_sources)
    dr_targets = 10 ** rng.uniform(0.0, np.log10(dr_max + EPS), size=num_sources)
    amplitudes = base_amplitudes * dr_targets / max(dr_targets.max(), EPS)

    for idx in range(num_sources):
        activity = make_activity_mask(n_samples, block, rng, activity_prob)
        signal_kind = rng.choice(["tone", "chirp", "burst"])
        phase = rng.uniform(0, 2 * np.pi)

        if signal_kind == "tone":
            freq = rng.uniform(5.0, 80.0)
            base = np.sin(2 * np.pi * freq * t + phase)
        elif signal_kind == "chirp":
            f0 = rng.uniform(4.0, 20.0)
            f1 = rng.uniform(40.0, 120.0)
            base = chirp(t, f0=f0, f1=f1, t1=t[-1], method="linear", phi=phase * 180 / np.pi)
        else:  # bursty noise
            base = rng.normal(scale=1.0, size=n_samples)
            sos = signal.butter(4, [4.0, 60.0], btype="bandpass", fs=fs, output="sos")
            base = signal.sosfiltfilt(sos, base)

        env = signal.windows.tukey(n_samples, alpha=0.1)
        signal_vec = amplitudes[idx] * base * activity * env
        signals.append(signal_vec)

        metadata.append(
            {
                "kind": signal_kind,
                "amplitude": float(amplitudes[idx]),
                "activity_fraction": float(activity.mean()),
            }
        )

    return np.vstack(signals), metadata


def run_ubss(B: np.ndarray, config: SimulationConfig) -> np.ndarray:
    UBSS.fs = config.sample_rate
    UBSS.bpo = config.ubss_bpo
    UBSS.sigma = config.ubss_sigma
    UBSS.lambda_ = config.ubss_lambda
    UBSS.sspTol = config.ubss_ssp_tol
    UBSS.cs_iters = config.ubss_cs_iters
    UBSS.detrend = False
    UBSS.boom = None

    estimate = UBSS.clean(np.asarray(B, dtype=float), triaxial=False)
    return np.squeeze(np.asarray(estimate))


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

    return stacked, subband_lengths, nsgt_obj


def flatten_tf(matrix: np.ndarray) -> np.ndarray:
    return matrix.reshape(matrix.shape[0], -1)


def wrap_phase(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def build_per_bin_table(
    seed: int,
    ambient_tf: np.ndarray,
    ambient_hat_tf: np.ndarray,
    mixture_tf: np.ndarray,
    interference_tf: np.ndarray,
    source_tf: np.ndarray,
    A: np.ndarray,
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

    source_magnitudes = np.abs(source_flat)
    if source_magnitudes.size:
        active_counts = np.sum(source_magnitudes > config.source_activity_threshold, axis=0)
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
        active_counts = np.zeros(n_bins)
        dr_proxy = np.ones(n_bins)

    rho_k = active_counts / max(A.shape[0], 1)

    mu = mutual_coherence(A)
    cond_val = (
        float(np.linalg.cond(A.T @ A + EPS * np.eye(A.shape[1]))) if A.size else 1.0
    )

    delta_vals: List[float] = []
    for idx in range(n_bins):
        if source_flat.shape[0] == 0 or np.linalg.norm(source_flat[:, idx]) < EPS or A.size == 0:
            delta_vals.append(0.0)
        else:
            delta_vals.append(float(calculate_delta_s(A, source_flat[:, idx])))

    mag_mask = np.linalg.norm(mixture_flat, axis=0) > config.magnitude_mask_threshold
    ssp_mask = active_counts <= 1

    records: List[Dict[str, object]] = []
    for idx in range(n_bins):
        records.append(
            {
                "seed": seed,
                "bin_index": idx,
                "ambient_truth": ambient_flat[idx],
                "ambient_estimate": ambient_hat_flat[idx],
                "mixture_vector": mixture_flat[:, idx],
                "magnitude_mask": bool(mag_mask[idx]),
                "ssp_mask": bool(ssp_mask[idx]),
                "cluster_label": -1,
                "A_id": 0,
                "mu_A": mu,
                "cond_AHA": cond_val,
                "delta_s": delta_vals[idx],
                "SNR_bin": snr_bins[idx],
                "MagRE_bin": mag_re[idx],
                "phase_error": phase_err[idx],
                "IAR_bin": iar_bins[idx],
                "DR_proxy_bin": dr_proxy[idx],
                "rho_k": rho_k[idx],
            }
        )

    return pd.DataFrame.from_records(records)


def simulate_seed(seed: int, config: SimulationConfig) -> SceneResult:
    rng = np.random.default_rng(seed)
    n_samples = int(config.sample_rate * config.duration_s)

    sensors = make_sensors(config.num_sensors, rng)
    sources = make_sources(config.num_sources, rng)
    mixing_matrix = compute_mixing_matrix(sensors, sources, config.axis)

    ambient = generate_ambient_signal(n_samples, config.sample_rate, rng, config.ambient_amp_range)
    ambient_per_sensor = np.outer(np.ones(config.num_sensors), ambient)

    source_signals, metadata = generate_interference_signals(
        config.num_sources,
        n_samples,
        config.sample_rate,
        rng,
        config.interference_amp_range,
        config.interference_dr_max,
        config.source_activity_prob,
    )

    interference = mixing_matrix @ source_signals
    mixture = ambient_per_sensor + interference
    if config.noise_std > 0:
        mixture += rng.normal(scale=config.noise_std, size=mixture.shape)

    ambient_hat = run_ubss(mixture, config)
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
        mixing_matrix,
        config,
    )

    weights = np.abs(ambient_tf.flatten()) ** 2
    if weights.sum() > 0:
        metrics["SNR_out_energy_weighted"] = float(np.sum(weights * per_bin["SNR_bin"]) / np.sum(weights))
    else:
        metrics["SNR_out_energy_weighted"] = metrics["SNR_out"]

    metrics.update(
        {
            "seed": seed,
            "mu_A": mutual_coherence(mixing_matrix),
            "cond_AHA": float(np.linalg.cond(mixing_matrix.T @ mixing_matrix + EPS * np.eye(config.num_sources)))
            if mixing_matrix.size
            else 1.0,
            "rho_k_mean": float(per_bin["rho_k"].mean()),
            "rho_k_median": float(per_bin["rho_k"].median()),
            "rho_k_p90": float(per_bin["rho_k"].quantile(0.9)),
            "phase_error_median": float(per_bin["phase_error"].median()),
            "phase_error_p90": float(per_bin["phase_error"].quantile(0.9)),
            "ssp_fraction": float(per_bin["ssp_mask"].mean()),
        }
    )

    return SceneResult(seed=seed, per_bin=per_bin, scene_metrics=metrics)


def fit_log_log_model(per_scene: pd.DataFrame, eps: float = 1e-9) -> Dict[str, float]:
    columns = ["nRMSE", "IAR_scene", "DR_scene", "mu_A", "rho_k_median"]
    df = per_scene[columns].replace([np.inf, -np.inf], np.nan).dropna()
    mask = (
        (df["nRMSE"] > 0)
        & (df["IAR_scene"] > 0)
        & (df["DR_scene"] > 0)
        & (df["mu_A"] > 0)
        & (df["rho_k_median"] > 0)
    )
    df = df[mask]

    if len(df) < 2:
        return {"status": "insufficient_data", "samples": int(len(df))}

    y = np.log(df["nRMSE"].to_numpy())
    log_IAR = np.log(df["IAR_scene"].to_numpy() + eps)
    log_DR = np.log(df["DR_scene"].to_numpy() + eps)
    log_mu = np.log(df["mu_A"].to_numpy() + eps)
    log_rho = np.log(df["rho_k_median"].to_numpy() + eps)

    X = np.column_stack([np.ones(len(df)), log_IAR, log_DR, log_mu, log_rho])
    coeffs, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    preds = X @ coeffs
    residual_vec = y - preds
    dof = max(len(df) - X.shape[1], 1)
    if residuals.size > 0:
        sigma2 = residuals[0] / dof
    else:
        sigma2 = float((residual_vec @ residual_vec) / dof)

    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(XtX_inv * sigma2))

    ss_tot = np.sum((y - y.mean()) ** 2)
    ss_res = np.sum(residual_vec**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {
        "status": "ok",
        "samples": int(len(df)),
        "c0": float(coeffs[0]),
        "alpha": float(-coeffs[1]),
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


def _delta_snr_model(x, ds_max, I0, p, gamma, eta, eps0):
    IAR, mu, rho = x
    term = 1.0 - 1.0 / (1.0 + (IAR / (I0 + EPS)) ** p)
    return ds_max * term - gamma * np.log(mu + EPS) - eta * np.log(rho + EPS) + eps0


def fit_saturating_model(per_scene: pd.DataFrame) -> Dict[str, float]:
    columns = ["Delta_SNR", "IAR_scene", "mu_A", "rho_k_median"]
    df = per_scene[columns].replace([np.inf, -np.inf], np.nan).dropna()
    mask = (df["IAR_scene"] > 0) & (df["mu_A"] > 0) & (df["rho_k_median"] > 0)
    df = df[mask]

    if len(df) < 6:
        return {"status": "insufficient_data", "samples": int(len(df))}

    IAR = df["IAR_scene"].to_numpy()
    mu = df["mu_A"].to_numpy()
    rho = df["rho_k_median"].to_numpy()
    y = df["Delta_SNR"].to_numpy()

    initial = [max(y.max(), 0.1), np.median(IAR), 1.0, 0.5, 0.5, 0.0]
    bounds = ([0.0, EPS, 0.1, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, 5.0, np.inf, np.inf, np.inf])

    try:
        popt, pcov = curve_fit(
            _delta_snr_model,
            (IAR, mu, rho),
            y,
            p0=initial,
            bounds=bounds,
            maxfev=20000,
        )
        perr = np.sqrt(np.diag(pcov))
        return {
            "status": "ok",
            "samples": int(len(df)),
            "Delta_SNR_max": float(popt[0]),
            "I0": float(popt[1]),
            "p": float(popt[2]),
            "gamma": float(popt[3]),
            "eta": float(popt[4]),
            "epsilon0": float(popt[5]),
            "Delta_SNR_max_se": float(perr[0]),
            "I0_se": float(perr[1]),
            "p_se": float(perr[2]),
            "gamma_se": float(perr[3]),
            "eta_se": float(perr[4]),
            "epsilon0_se": float(perr[5]),
        }
    except Exception as exc:  # pragma: no cover - debug aid
        return {
            "status": "fit_failed",
            "samples": int(len(df)),
            "message": str(exc),
        }


def fit_scaling_models(per_scene: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    return {
        "log_model": fit_log_log_model(per_scene),
        "saturating_model": fit_saturating_model(per_scene),
    }


def run_simulation(config: SimulationConfig) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]]]:
    scene_results: List[SceneResult] = [simulate_seed(seed, config) for seed in config.seeds]

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

    fit_results = fit_scaling_models(per_scene)

    return per_bin, per_scene, fit_results


def run_small_demo() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]]]:
    config = SimulationConfig()
    return run_simulation(config)


if __name__ == "__main__":
    per_bin_df, scene_df, fits = run_small_demo()
    print("Per-scene summary:\n", scene_df.head())
    print("Per-bin preview:\n", per_bin_df.head())
    print("Log-log fit:", fits["log_model"])
    print("Saturating fit:", fits["saturating_model"])
