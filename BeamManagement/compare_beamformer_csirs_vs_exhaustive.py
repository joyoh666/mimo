"""
Compare BeamFormer CSI-RS beam selection against a same-overhead beam sweep.

This is a standalone experiment script. It does not modify the BeamFormer
package. By default, the sweep baseline uses 64 uniformly subsampled beams from
the same 80 x 20 query grid that BeamFormer predicts over. This gives the
baseline the same number of measurements as BeamFormer's 64 CSI-RS beams.

Example:
    python compare_beamformer_csirs_vs_exhaustive.py \
        --data_path BeamFormer/csi-dataset/homeoffice-communication-28G-csi/t16x16_r2x1_test_small \
        --model_dir BeamFormer/saved_models/co_train \
        --max_samples 128
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR if (SCRIPT_DIR / "BeamFormer").exists() else SCRIPT_DIR.parent
BEAMFORMER_ROOT = PROJECT_ROOT / "BeamFormer"
if str(BEAMFORMER_ROOT) not in sys.path:
    sys.path.insert(0, str(BEAMFORMER_ROOT))


DEFAULT_DATA_CANDIDATES = (
    BEAMFORMER_ROOT
    / "csi-dataset"
    / "homeoffice-communication-28G-csi"
    / "t16x16_r2x1_test_small",
    BEAMFORMER_ROOT / "mini_demo" / "indoor_28g_dataset" / "t16x16_r2x1_test_small",
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "comparison_results"


def import_runtime_modules():
    global ARN_model
    global InferenceHelper
    global assumption
    global dataset
    global estimator
    global generator
    global get_db
    global np
    global plt
    global read_csi_file_to_torch
    global torch
    global tqdm
    global transform_weights

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        import numpy as _np
        import torch as _torch
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Missing runtime dependency. Install BeamFormer requirements first, "
            "for example: pip install -r BeamFormer/requirements.txt"
        ) from error

    try:
        from tqdm import tqdm as _tqdm
    except ModuleNotFoundError:
        def _tqdm(iterable, **_kwargs):
            return iterable

    try:
        from beamformer.inference_helper import InferenceHelper as _InferenceHelper
        from beamformer.utils import get_db as _get_db
        from beamformer.utils import read_csi_file_to_torch as _read_csi_file_to_torch
        from beamformer.weight_generator import transform_weights as _transform_weights
        from configs.submodules import ARN_model as _ARN_model
        from configs.submodules import assumption as _assumption
        from configs.submodules import dataset as _dataset
        from configs.submodules import estimator as _estimator
        from configs.submodules import generator as _generator
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Missing BeamFormer dependency. Install BeamFormer requirements first, "
            "for example: pip install -r BeamFormer/requirements.txt"
        ) from error

    ARN_model = _ARN_model
    InferenceHelper = _InferenceHelper
    assumption = _assumption
    dataset = _dataset
    estimator = _estimator
    generator = _generator
    get_db = _get_db
    np = _np
    plt = _plt
    read_csi_file_to_torch = _read_csi_file_to_torch
    torch = _torch
    tqdm = _tqdm
    transform_weights = _transform_weights


def parse_ebno_dbs(value: str) -> np.ndarray:
    return np.array([float(item.strip()) for item in value.split(",") if item.strip()])


def resolve_path(value: str | None, *, must_exist: bool = False) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def first_existing(paths: tuple[Path, ...]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def resolve_model_file(
    explicit_path: str | None,
    model_dir: Path | None,
    candidate_names: tuple[str, ...],
    label: str,
) -> Path:
    if explicit_path is not None:
        return resolve_path(explicit_path, must_exist=True)

    search_dirs = []
    if model_dir is not None:
        search_dirs.append(model_dir)
    search_dirs.extend(
        [
            BEAMFORMER_ROOT / "saved_models" / "co_train",
            BEAMFORMER_ROOT / "mini_demo" / "saved_models",
        ]
    )

    for directory in search_dirs:
        for name in candidate_names:
            path = directory / name
            if path.exists():
                return path.resolve()

    searched = ", ".join(str(directory / name) for directory in search_dirs for name in candidate_names)
    raise FileNotFoundError(f"Could not find {label}. Searched: {searched}")


def resolve_optional_model_file(
    explicit_path: str | None,
    model_dir: Path | None,
    candidate_names: tuple[str, ...],
) -> Path | None:
    if explicit_path is not None:
        return resolve_path(explicit_path, must_exist=True)

    search_dirs = []
    if model_dir is not None:
        search_dirs.append(model_dir)
    search_dirs.extend(
        [
            BEAMFORMER_ROOT / "ARN_saved_models" / "arn",
            BEAMFORMER_ROOT / "mini_demo" / "saved_models",
        ]
    )
    for directory in search_dirs:
        for name in candidate_names:
            path = directory / name
            if path.exists():
                return path.resolve()
    return None


def build_setting(data_path: Path, generator_path: Path, estimator_path: Path, arn_path: Path | None):
    ds = dataset.homeoffice_communication_28g()
    ds.test_data_path = str(data_path)

    return SimpleNamespace(
        name="beamformer_csirs_vs_exhaustive",
        dataset=ds,
        assumption=assumption.beam64(),
        scheme="co-train",
        generator=generator.parametric_generator(
            generator_pretrained_model=str(generator_path)
        ),
        estimator=estimator.PerceiverIO(
            estimator_pretrained_model=str(estimator_path)
        ),
        arn_model=ARN_model.typical_ARN(
            ARN_model_pretrained_model=str(arn_path) if arn_path is not None else None
        ),
    )


def make_reference_weights(helper: InferenceHelper) -> torch.Tensor:
    setting = helper.setting
    z_dim = setting.assumption.sample_num * setting.dataset.M * setting.dataset.N
    z = torch.zeros(1, z_dim, device=helper.device)
    raw_weights = helper.generator(z)
    weights, _ = transform_weights(raw_weights)
    return weights.detach().cpu()


def simulate_csirs_rss(helper: InferenceHelper, csi_tensor: torch.Tensor, reference_weights: torch.Tensor):
    if csi_tensor.ndim == 3:
        csi_tensor = csi_tensor.unsqueeze(0)
    csi_tensor = csi_tensor.to(helper.device)
    weights = reference_weights.to(helper.device)
    return helper.dp.generate_sample_rss(csi_tensor, weights)[0].detach().cpu()


def predict_beamformer_beam(
    helper: InferenceHelper,
    sample_rss: torch.Tensor,
    reference_weights: torch.Tensor,
    query_weights: torch.Tensor,
):
    scale = torch.clamp(torch.max(sample_rss), min=1e-12)
    sample_rss_norm = (sample_rss / scale).to(dtype=torch.float32)

    pred_spectrum = helper.infer_from_rss(sample_rss_norm, reference_weights)
    pred_spectrum = pred_spectrum.reshape(-1) * scale
    best_index = int(torch.argmax(pred_spectrum).item())

    return {
        "index": best_index,
        "spectrum": pred_spectrum.detach().cpu(),
        "beam": query_weights[0, best_index].detach().cpu(),
        "score": float(pred_spectrum[best_index].item()),
    }


def choose_query_grid_counts(sweep_size: int, phi_steps: int, theta_steps: int):
    """Choose a near-uniform phi/theta factorization for sparse sweeping."""
    if sweep_size < 1:
        raise ValueError("sweep_size must be positive.")

    sweep_size = min(sweep_size, phi_steps * theta_steps)
    target_ratio = phi_steps / theta_steps
    best = None
    for theta_count in range(1, min(theta_steps, sweep_size) + 1):
        phi_count = min(phi_steps, sweep_size // theta_count)
        if phi_count < 1:
            continue
        count = phi_count * theta_count
        ratio = phi_count / theta_count
        score = (
            sweep_size - count,
            abs(math.log(max(ratio, 1e-12) / target_ratio)),
            -theta_count,
        )
        if best is None or score < best[0]:
            best = (score, phi_count, theta_count)

    if best is None:
        raise ValueError("Could not build sparse query-grid sweep.")
    return best[1], best[2]


def evenly_spaced_int_indices(num_bins: int, count: int):
    if count < 1 or count > num_bins:
        raise ValueError(f"count must be in [1, {num_bins}], got {count}.")
    return np.rint(np.linspace(0, num_bins - 1, count)).astype(np.int64)


def make_query_grid_candidate_indices(
    helper: InferenceHelper,
    sweep_size: int,
    sweep_mode: str,
    seed: int,
):
    theta_steps = helper.setting.assumption.angle_steps_theta
    phi_steps = helper.setting.assumption.angle_steps_phi
    query_num = theta_steps * phi_steps

    if sweep_mode == "full_query_grid":
        return torch.arange(query_num, dtype=torch.long), f"{query_num}-beam full query-grid sweep"

    if sweep_mode == "random_query_grid":
        rng = np.random.default_rng(seed)
        candidate_count = min(sweep_size, query_num)
        indices = np.sort(rng.choice(query_num, size=candidate_count, replace=False))
        return torch.from_numpy(indices).to(dtype=torch.long), f"{candidate_count}-beam random query-grid sweep"

    if sweep_mode != "uniform_query_grid":
        raise ValueError(f"Unsupported sweep_mode: {sweep_mode}")

    phi_count, theta_count = choose_query_grid_counts(sweep_size, phi_steps, theta_steps)
    phi_indices = evenly_spaced_int_indices(phi_steps, phi_count)
    theta_indices = evenly_spaced_int_indices(theta_steps, theta_count)
    flat_indices = [
        int(phi_idx * theta_steps + theta_idx)
        for phi_idx in phi_indices
        for theta_idx in theta_indices
    ]
    label = f"{len(flat_indices)}-beam uniform query-grid sweep ({phi_count} x {theta_count})"
    return torch.tensor(flat_indices, dtype=torch.long), label


def select_sweep_beam(
    helper: InferenceHelper,
    csi_tensor: torch.Tensor,
    query_weights: torch.Tensor,
    candidate_indices: torch.Tensor,
):
    if csi_tensor.ndim == 3:
        csi_tensor = csi_tensor.unsqueeze(0)
    candidate_indices = candidate_indices.to(dtype=torch.long)
    candidate_weights = query_weights[0, candidate_indices]
    candidate_rss = measure_beam_rss(helper, csi_tensor, candidate_weights)
    best_local_index = int(torch.argmax(candidate_rss).item())
    best_index = int(candidate_indices[best_local_index].item())
    return {
        "index": best_index,
        "candidate_indices": candidate_indices.detach().cpu(),
        "candidate_rss": candidate_rss.detach().cpu(),
        "beam": query_weights[0, best_index].detach().cpu(),
        "score": float(candidate_rss[best_local_index].item()),
    }


def measure_beam_rss(helper: InferenceHelper, csi_tensor: torch.Tensor, beam_weights: torch.Tensor):
    csi = csi_tensor.to(helper.device)
    if csi.ndim == 3:
        csi = csi.unsqueeze(0)

    weights = torch.as_tensor(beam_weights).to(helper.device)
    if weights.ndim == 2:
        weights = weights.unsqueeze(0)
    if weights.ndim != 3:
        raise ValueError("beam_weights must have shape [M, N] or [num_beams, M, N].")

    weights = weights.to(dtype=csi.dtype).reshape(1, weights.shape[0], -1)
    return helper.dp._generate_rss(csi, weights)[0].detach().cpu()


def effective_channels(helper: InferenceHelper, csi_tensor: torch.Tensor, beam_weights: torch.Tensor):
    csi = csi_tensor.to(helper.device)
    if csi.ndim == 3:
        csi = csi.unsqueeze(0)
    if csi.shape[0] != 1:
        raise ValueError("effective_channels expects one CSI sample.")

    weights = torch.as_tensor(beam_weights).to(helper.device)
    if weights.ndim == 2:
        weights = weights.unsqueeze(0)
    if weights.ndim != 3:
        raise ValueError("beam_weights must have shape [M, N] or [num_beams, M, N].")

    weights = weights.to(dtype=csi.dtype).reshape(weights.shape[0], -1)
    h_eff = torch.einsum("frt,bt->bfr", csi[0], weights)
    return h_eff[..., 0].detach().cpu()


def optimal_beam_from_csi(helper: InferenceHelper, csi_tensor: torch.Tensor):
    """Compute the unconstrained unit-norm beam that maximizes average RSS.

    For the default BeamFormer setting, Tx is beamformed and the first Rx
    antenna is active. The optimal vector is the dominant eigenvector of the
    wideband channel covariance.
    """
    csi = csi_tensor.to(helper.device)
    if csi.ndim == 3:
        csi = csi.unsqueeze(0)
    if csi.shape[0] != 1:
        raise ValueError("optimal_beam_from_csi expects one CSI sample.")

    if helper.setting.dataset.mode == "rx_act1":
        channel = csi[0, :, 0, :]
    elif helper.setting.dataset.mode == "tx_act1":
        channel = csi[0, :, :, 0]
    else:
        raise ValueError(f"Unsupported dataset mode: {helper.setting.dataset.mode}")

    covariance = channel.conj().transpose(0, 1) @ channel
    covariance = covariance / max(channel.shape[0], 1)
    _, eigenvectors = torch.linalg.eigh(covariance)
    beam = eigenvectors[:, -1]
    beam = beam / torch.clamp(torch.linalg.vector_norm(beam), min=1e-12)
    return beam.reshape(helper.setting.dataset.M, helper.setting.dataset.N).detach().cpu()


def beam_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = torch.as_tensor(a, dtype=torch.complex64).reshape(-1)
    b = torch.as_tensor(b, dtype=torch.complex64).reshape(-1)
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if denom <= 0:
        return 0.0
    return float(torch.abs(torch.vdot(a, b)) / denom)


def index_to_angles(helper: InferenceHelper, flat_index: int):
    theta_steps = helper.setting.assumption.angle_steps_theta
    phi_steps = helper.setting.assumption.angle_steps_phi
    phi_index = flat_index // theta_steps
    theta_index = flat_index % theta_steps
    phi_deg = phi_index * (360.0 / phi_steps)
    theta_deg = theta_index * (helper.setting.dataset.max_theta / (theta_steps - 1))
    return phi_index, theta_index, phi_deg, theta_deg


def wrapped_phi_error_deg(a: float, b: float) -> float:
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def qpsk_ber_from_effective_channels(
    ebno_dbs: np.ndarray,
    h_eff: torch.Tensor,
    symbols_per_channel: int,
    chunk_size: int,
    seed: int,
):
    h_eff = torch.as_tensor(h_eff, dtype=torch.complex64).reshape(-1)
    h_eff = h_eff[torch.abs(h_eff) > 1e-12]
    if h_eff.numel() == 0:
        return [1.0 for _ in ebno_dbs]

    rng = torch.Generator(device="cpu")
    ber = []
    sqrt_two = math.sqrt(2.0)

    for ebno_db in ebno_dbs:
        rng.manual_seed(seed + int(round(float(ebno_db) * 100)))
        ebno_linear = 10 ** (float(ebno_db) / 10.0)
        noise_variance = 1.0 / (2.0 * ebno_linear)
        noise_std = math.sqrt(noise_variance / 2.0)

        bit_errors = 0
        total_bits = 0

        for start in range(0, h_eff.numel(), chunk_size):
            h_chunk = h_eff[start : start + chunk_size]
            bits = torch.randint(
                0,
                2,
                (h_chunk.numel(), symbols_per_channel, 2),
                generator=rng,
                dtype=torch.int8,
            )
            symbols = (
                (1.0 - 2.0 * bits[..., 0].to(torch.float32))
                + 1j * (1.0 - 2.0 * bits[..., 1].to(torch.float32))
            ) / sqrt_two

            noise = noise_std * (
                torch.randn(symbols.shape, generator=rng)
                + 1j * torch.randn(symbols.shape, generator=rng)
            )
            y = h_chunk[:, None] * symbols + noise
            x_hat = y / h_chunk[:, None]

            bits_hat = torch.empty_like(bits)
            bits_hat[..., 0] = (x_hat.real < 0).to(torch.int8)
            bits_hat[..., 1] = (x_hat.imag < 0).to(torch.int8)

            bit_errors += int(torch.count_nonzero(bits_hat != bits).item())
            total_bits += bits.numel()

        ber.append(bit_errors / total_bits)

    return ber


def qpsk_ber_from_gain_ratios(ebno_dbs: np.ndarray, gain_ratios):
    """Uncoded QPSK BER from beamforming-gain ratios.

    The optimal beam is the reference with gain ratio 1.0. Other beams get an
    effective Eb/N0 scaled by measured_rss / optimal_rss.
    """
    ratios = np.asarray(gain_ratios, dtype=np.float64)
    ratios = np.maximum(ratios, 1e-16)
    ber = []
    for ebno_db in ebno_dbs:
        ebno_linear = 10 ** (float(ebno_db) / 10.0)
        sample_ber = [0.5 * math.erfc(math.sqrt(ebno_linear * ratio)) for ratio in ratios]
        ber.append(float(np.mean(sample_ber)))
    return ber


def qpsk_ber_monte_carlo_from_gain_ratios(
    ebno_dbs: np.ndarray,
    gain_ratios,
    symbols_per_sample: int,
    seed: int,
):
    """Monte Carlo uncoded QPSK BER from beamforming-gain ratios.

    The optimal beam has gain ratio 1.0, so Eb/N0 is interpreted as the
    optimal-beam Eb/N0. Each selected beam uses effective Eb/N0 =
    Eb/N0 * selected_rss / optimal_rss.
    """
    ratios = torch.as_tensor(gain_ratios, dtype=torch.float32).reshape(-1)
    ratios = torch.clamp(ratios, min=1e-16)
    if ratios.numel() == 0:
        return [1.0 for _ in ebno_dbs]

    rng = torch.Generator(device="cpu")
    ber = []
    sqrt_two = math.sqrt(2.0)

    for ebno_db in ebno_dbs:
        rng.manual_seed(seed + int(round(float(ebno_db) * 100)))
        ebno_linear = 10 ** (float(ebno_db) / 10.0)
        effective_ebno = ebno_linear * ratios
        noise_std = torch.sqrt(1.0 / (2.0 * effective_ebno))

        bits = torch.randint(
            0,
            2,
            (ratios.numel(), symbols_per_sample, 2),
            generator=rng,
            dtype=torch.int8,
        )
        symbols = (
            (1.0 - 2.0 * bits[..., 0].to(torch.float32))
            + 1j * (1.0 - 2.0 * bits[..., 1].to(torch.float32))
        ) / sqrt_two

        noise = (noise_std[:, None] / sqrt_two) * (
            torch.randn(symbols.shape, generator=rng)
            + 1j * torch.randn(symbols.shape, generator=rng)
        )
        received = symbols + noise

        bits_hat = torch.empty_like(bits)
        bits_hat[..., 0] = (received.real < 0).to(torch.int8)
        bits_hat[..., 1] = (received.imag < 0).to(torch.int8)

        bit_errors = int(torch.count_nonzero(bits_hat != bits).item())
        total_bits = bits.numel()
        ber.append(bit_errors / total_bits)

    return ber


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.percentile(values, 50)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def collect_files(data_path: Path, max_samples: int | None):
    files = sorted(data_path.glob("*.mat"))
    if max_samples is not None:
        files = files[:max_samples]
    if not files:
        raise FileNotFoundError(f"No .mat files found in {data_path}")
    return files


def write_csv(rows, output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload, output_json: Path):
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as handle:
        json.dump(payload, handle, indent=2)


def plot_ber(ebno_dbs, ber_curves, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=180)
    markers = ["o", "s", "^", "D"]
    for marker, (label, ber) in zip(markers, ber_curves.items()):
        ax.semilogy(ebno_dbs, ber, marker=marker, linewidth=2, label=label)
    ax.grid(True, which="both", linestyle="--", alpha=0.45)
    ax.set_xlabel("Eb/N0 [dB]")
    ax.set_ylabel("BER")
    ax.set_title("BER from Beamforming Gain Relative to Optimal")
    ax.set_ylim(1e-6, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_cdf(values, output_path: Path, xlabel: str, title: str, xlim=None):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = np.sort(np.asarray(values, dtype=np.float64))
    cdf = np.arange(1, len(values) + 1) / len(values)
    x = np.r_[values[0], values]
    y = np.r_[0.0, cdf]
    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=180)
    ax.step(x, y, where="post", linewidth=2)
    ax.scatter(values, cdf, s=10, zorder=3)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_multi_cdf(series, output_path: Path, xlabel: str, title: str, xlim=None):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=180)
    for label, raw_values in series.items():
        values = np.sort(np.asarray(raw_values, dtype=np.float64))
        cdf = np.arange(1, len(values) + 1) / len(values)
        x = np.r_[values[0], values]
        y = np.r_[0.0, cdf]
        ax.step(x, y, where="post", linewidth=2, label=label)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def evaluate(
    helper: InferenceHelper,
    files,
    include_best_reference: bool,
    sweep_size: int,
    sweep_mode: str,
    seed: int,
):
    reference_weights = make_reference_weights(helper)
    query_weights = helper.dp.generate_query_weights(batch_size=1).detach().cpu()
    candidate_indices, sweep_label = make_query_grid_candidate_indices(
        helper, sweep_size=sweep_size, sweep_mode=sweep_mode, seed=seed
    )

    rows = []
    gain_ratio_sets = {
        "Optimal": [],
        sweep_label: [],
        "BeamFormer CSI-RS": [],
    }
    if include_best_reference:
        gain_ratio_sets["Best measured CSI-RS ref"] = []

    for path in tqdm(files, desc="[Compare]"):
        csi = read_csi_file_to_torch(str(path)).unsqueeze(0)

        optimal_beam = optimal_beam_from_csi(helper, csi)
        sample_rss = simulate_csirs_rss(helper, csi, reference_weights)
        beamformer = predict_beamformer_beam(helper, sample_rss, reference_weights, query_weights)
        sweep = select_sweep_beam(helper, csi, query_weights, candidate_indices)

        candidate_beams = [optimal_beam, sweep["beam"], beamformer["beam"]]

        best_reference_index = int(torch.argmax(sample_rss).item())
        if include_best_reference:
            candidate_beams.append(reference_weights[0, best_reference_index])

        candidate_beams = torch.stack(candidate_beams, dim=0)
        actual_rss = measure_beam_rss(helper, csi, candidate_beams)

        optimal_rss = max(float(actual_rss[0].item()), 1e-16)
        sweep_rss = max(float(actual_rss[1].item()), 1e-16)
        beamformer_rss = max(float(actual_rss[2].item()), 1e-16)

        gain_ratio_sets["Optimal"].append(1.0)
        gain_ratio_sets[sweep_label].append(sweep_rss / optimal_rss)
        gain_ratio_sets["BeamFormer CSI-RS"].append(beamformer_rss / optimal_rss)

        bf_phi_idx, bf_theta_idx, bf_phi_deg, bf_theta_deg = index_to_angles(
            helper, beamformer["index"]
        )
        sweep_phi_idx, sweep_theta_idx, sweep_phi_deg, sweep_theta_deg = index_to_angles(
            helper, sweep["index"]
        )

        sweep_similarity = beam_similarity(optimal_beam, sweep["beam"])
        beamformer_similarity = beam_similarity(optimal_beam, beamformer["beam"])
        sweep_gain_loss_db = float(get_db(optimal_rss) - get_db(sweep_rss))
        beamformer_gain_loss_db = float(get_db(optimal_rss) - get_db(beamformer_rss))

        row = {
            "file": path.name,
            "beamformer_index": beamformer["index"],
            "beamformer_phi_index": bf_phi_idx,
            "beamformer_theta_index": bf_theta_idx,
            "beamformer_phi_deg": bf_phi_deg,
            "beamformer_theta_deg": bf_theta_deg,
            "sweep_index": sweep["index"],
            "sweep_phi_index": sweep_phi_idx,
            "sweep_theta_index": sweep_theta_idx,
            "sweep_phi_deg": sweep_phi_deg,
            "sweep_theta_deg": sweep_theta_deg,
            "beamformer_matches_sweep": beamformer["index"] == sweep["index"],
            "sweep_similarity_to_optimal": sweep_similarity,
            "beamformer_similarity_to_optimal": beamformer_similarity,
            "sweep_similarity_squared_to_optimal": sweep_similarity**2,
            "beamformer_similarity_squared_to_optimal": beamformer_similarity**2,
            "beamformer_vs_sweep_phi_error_deg": wrapped_phi_error_deg(bf_phi_deg, sweep_phi_deg),
            "beamformer_vs_sweep_theta_error_deg": abs(bf_theta_deg - sweep_theta_deg),
            "optimal_rss": optimal_rss,
            "beamformer_rss": beamformer_rss,
            "sweep_rss": sweep_rss,
            "sweep_gain_loss_to_optimal_db": sweep_gain_loss_db,
            "beamformer_gain_loss_to_optimal_db": beamformer_gain_loss_db,
            "best_reference_index": best_reference_index,
        }
        if include_best_reference:
            ref_rss = max(float(actual_rss[3].item()), 1e-16)
            ref_similarity = beam_similarity(optimal_beam, reference_weights[0, best_reference_index])
            gain_ratio_sets["Best measured CSI-RS ref"].append(ref_rss / optimal_rss)
            row["best_reference_rss"] = ref_rss
            row["best_reference_similarity_to_optimal"] = ref_similarity
            row["best_reference_loss_to_optimal_db"] = float(get_db(optimal_rss) - get_db(ref_rss))

        rows.append(row)

    return rows, gain_ratio_sets, sweep_label, len(candidate_indices)


def main():
    parser = argparse.ArgumentParser(
        description="Compare BeamFormer CSI-RS selection with same-overhead query-grid sweeping."
    )
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--generator_path", type=str, default=None)
    parser.add_argument("--estimator_path", type=str, default=None)
    parser.add_argument("--arn_model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--sweep_size",
        type=int,
        default=64,
        help="Number of query-grid beams measured by the sparse sweep baseline.",
    )
    parser.add_argument(
        "--sweep_mode",
        type=str,
        default="uniform_query_grid",
        choices=["uniform_query_grid", "random_query_grid", "full_query_grid"],
        help=(
            "How to choose sparse sweep beams. uniform_query_grid gives the "
            "recommended deterministic 64-beam baseline."
        ),
    )
    parser.add_argument("--ebno_dbs", type=str, default="-5,0,5,10,15,20")
    parser.add_argument(
        "--ber_mode",
        type=str,
        default="monte_carlo",
        choices=["monte_carlo", "analytic"],
        help="BER estimation method. Monte Carlo simulates QPSK symbols using gain-ratio-scaled Eb/N0.",
    )
    parser.add_argument("--ber_symbols_per_channel", type=int, default=512)
    parser.add_argument("--ber_chunk_size", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include_best_reference",
        action="store_true",
        help="Also plot the best of the 64 measured CSI-RS reference beams.",
    )
    args = parser.parse_args()

    import_runtime_modules()
    torch.set_grad_enabled(False)

    data_path = resolve_path(args.data_path) if args.data_path else first_existing(DEFAULT_DATA_CANDIDATES)
    if data_path is None or not data_path.exists():
        candidates = "\n  ".join(str(path) for path in DEFAULT_DATA_CANDIDATES)
        raise FileNotFoundError(
            "Could not find a CSI data directory. Pass --data_path explicitly. "
            f"Default candidates were:\n  {candidates}"
        )

    model_dir = resolve_path(args.model_dir) if args.model_dir else None
    generator_path = resolve_model_file(
        args.generator_path,
        model_dir,
        ("generator_epoch_final.pth", "generator.pth"),
        "generator weights",
    )
    estimator_path = resolve_model_file(
        args.estimator_path,
        model_dir,
        ("estimator_epoch_final.pth", "estimator.pth"),
        "estimator weights",
    )
    arn_path = resolve_optional_model_file(
        args.arn_model_path,
        model_dir,
        ("model_epoch_final.pth", "arn_model.pth"),
    )

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setting = build_setting(data_path, generator_path, estimator_path, arn_path)
    helper = InferenceHelper(setting)
    files = collect_files(data_path, args.max_samples)

    rows, gain_ratio_sets, sweep_label, actual_sweep_size = evaluate(
        helper,
        files,
        args.include_best_reference,
        sweep_size=args.sweep_size,
        sweep_mode=args.sweep_mode,
        seed=args.seed,
    )

    ebno_dbs = parse_ebno_dbs(args.ebno_dbs)
    if args.ber_mode == "monte_carlo":
        ber_curves = {
            label: qpsk_ber_monte_carlo_from_gain_ratios(
                ebno_dbs,
                ratios,
                symbols_per_sample=args.ber_symbols_per_channel,
                seed=args.seed,
            )
            for label, ratios in gain_ratio_sets.items()
        }
        ber_type = (
            "Monte Carlo uncoded QPSK using method RSS / optimal RSS as "
            "the Eb/N0 scale"
        )
    else:
        ber_curves = {
            label: qpsk_ber_from_gain_ratios(ebno_dbs, ratios)
            for label, ratios in gain_ratio_sets.items()
        }
        ber_type = (
            "Analytic uncoded QPSK using method RSS / optimal RSS as "
            "the Eb/N0 scale"
        )

    similarity_sets = {
        f"Optimal vs {sweep_label}": [
            row["sweep_similarity_to_optimal"] for row in rows
        ],
        "Optimal vs BeamFormer": [
            row["beamformer_similarity_to_optimal"] for row in rows
        ],
    }
    gain_loss_sets = {
        f"Optimal vs {sweep_label}": [
            row["sweep_gain_loss_to_optimal_db"] for row in rows
        ],
        "Optimal vs BeamFormer": [
            row["beamformer_gain_loss_to_optimal_db"] for row in rows
        ],
    }
    if args.include_best_reference:
        similarity_sets["Optimal vs best measured CSI-RS ref"] = [
            row["best_reference_similarity_to_optimal"] for row in rows
        ]
        gain_loss_sets["Optimal vs best measured CSI-RS ref"] = [
            row["best_reference_loss_to_optimal_db"] for row in rows
        ]

    all_gain_losses = [
        value for values in gain_loss_sets.values() for value in values
    ]
    gain_loss_xlim = (
        min(-0.05, float(np.min(all_gain_losses)) - 0.05),
        max(1.0, float(np.percentile(all_gain_losses, 99)) * 1.1),
    )

    summary = {
        "num_samples": len(rows),
        "data_path": str(data_path),
        "generator_path": str(generator_path),
        "estimator_path": str(estimator_path),
        "arn_model_path": str(arn_path) if arn_path is not None else None,
        "optimal_beam_definition": "dominant eigenvector of mean_f H_f^H H_f for the active receiver",
        "sweep_baseline": {
            "label": sweep_label,
            "mode": args.sweep_mode,
            "requested_sweep_size": args.sweep_size,
            "actual_sweep_size": actual_sweep_size,
            "query_grid": "80 x 20 BeamFormer query beam grid",
        },
        "beamformer_sweep_match_rate": float(
            np.mean([row["beamformer_matches_sweep"] for row in rows])
        ),
        "similarity_to_optimal": {
            label: summarize(values) for label, values in similarity_sets.items()
        },
        "gain_loss_to_optimal_db": {
            label: summarize(values) for label, values in gain_loss_sets.items()
        },
        "ebno_dbs": ebno_dbs.tolist(),
        "ber_mode": args.ber_mode,
        "ber_type": ber_type,
        "ber_symbols_per_sample": args.ber_symbols_per_channel,
        "ber": ber_curves,
    }

    write_csv(rows, output_dir / "per_sample_metrics.csv")
    write_json(summary, output_dir / "summary.json")
    plot_ber(ebno_dbs, ber_curves, output_dir / "ber_comparison.png")
    plot_cdf(
        similarity_sets[f"Optimal vs {sweep_label}"],
        output_dir / "similarity_optimal_vs_sweep_cdf.png",
        "Normalized beam-vector similarity",
        "Optimal vs Sparse-Sweep Beam Similarity",
        xlim=(-0.02, 1.02),
    )
    plot_cdf(
        similarity_sets["Optimal vs BeamFormer"],
        output_dir / "similarity_optimal_vs_beamformer_cdf.png",
        "Normalized beam-vector similarity",
        "Optimal vs BeamFormer Beam Similarity",
        xlim=(-0.02, 1.02),
    )
    plot_multi_cdf(
        similarity_sets,
        output_dir / "similarity_to_optimal_cdf.png",
        "Normalized beam-vector similarity",
        "Similarity to Optimal Beam",
        xlim=(-0.02, 1.02),
    )
    plot_cdf(
        gain_loss_sets[f"Optimal vs {sweep_label}"],
        output_dir / "gain_loss_optimal_vs_sweep_cdf.png",
        "Beamforming gain loss to optimal [dB]",
        "Sparse-Sweep Loss to Optimal Beam",
        xlim=gain_loss_xlim,
    )
    plot_cdf(
        gain_loss_sets["Optimal vs BeamFormer"],
        output_dir / "gain_loss_optimal_vs_beamformer_cdf.png",
        "Beamforming gain loss to optimal [dB]",
        "BeamFormer Loss to Optimal Beam",
        xlim=gain_loss_xlim,
    )
    plot_multi_cdf(
        gain_loss_sets,
        output_dir / "gain_loss_to_optimal_cdf.png",
        "Beamforming gain loss to optimal [dB]",
        "Gain Loss to Optimal Beam",
        xlim=gain_loss_xlim,
    )

    print(f"Wrote CSV:              {output_dir / 'per_sample_metrics.csv'}")
    print(f"Wrote summary:          {output_dir / 'summary.json'}")
    print(f"Wrote BER plot:         {output_dir / 'ber_comparison.png'}")
    print(f"Wrote similarity CDFs:  {output_dir / 'similarity_to_optimal_cdf.png'}")
    print(f"Wrote gain-loss CDFs:   {output_dir / 'gain_loss_to_optimal_cdf.png'}")
    print(f"Sweep baseline:         {sweep_label}")
    print(f"BF/sweep match rate:    {summary['beamformer_sweep_match_rate']:.4f}")
    print(
        "Mean BF similarity:     "
        f"{summary['similarity_to_optimal']['Optimal vs BeamFormer']['mean']:.4f}"
    )
    print(
        "Median BF loss [dB]:    "
        f"{summary['gain_loss_to_optimal_db']['Optimal vs BeamFormer']['median']:.4f}"
    )


if __name__ == "__main__":
    main()
