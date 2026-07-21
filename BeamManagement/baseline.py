"""Shared BeamFormer simulation baselines and experiment utilities.

This module contains the reusable runtime setup, channel/beam measurements,
optimal and query-grid sweep baselines, BER simulation, result serialization,
and plotting helpers used by the comparison scripts in this directory.
"""

from __future__ import annotations

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
DEFAULT_MONTE_CARLO_BITS = 1_000_000


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

    searched = ", ".join(
        str(directory / name)
        for directory in search_dirs
        for name in candidate_names
    )
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


def build_setting(
    data_path: Path,
    generator_path: Path,
    estimator_path: Path,
    arn_path: Path | None,
):
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


def simulate_csirs_rss(
    helper: InferenceHelper,
    csi_tensor: torch.Tensor,
    reference_weights: torch.Tensor,
):
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
        return (
            torch.arange(query_num, dtype=torch.long),
            f"{query_num}-beam full query-grid sweep",
        )

    if sweep_mode == "random_query_grid":
        rng = np.random.default_rng(seed)
        candidate_count = min(sweep_size, query_num)
        indices = np.sort(rng.choice(query_num, size=candidate_count, replace=False))
        return (
            torch.from_numpy(indices).to(dtype=torch.long),
            f"{candidate_count}-beam random query-grid sweep",
        )

    if sweep_mode != "uniform_query_grid":
        raise ValueError(f"Unsupported sweep_mode: {sweep_mode}")

    phi_count, theta_count = choose_query_grid_counts(
        sweep_size,
        phi_steps,
        theta_steps,
    )
    phi_indices = evenly_spaced_int_indices(phi_steps, phi_count)
    theta_indices = evenly_spaced_int_indices(theta_steps, theta_count)
    flat_indices = [
        int(phi_idx * theta_steps + theta_idx)
        for phi_idx in phi_indices
        for theta_idx in theta_indices
    ]
    label = (
        f"{len(flat_indices)}-beam uniform query-grid sweep "
        f"({phi_count} x {theta_count})"
    )
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


def measure_beam_rss(
    helper: InferenceHelper,
    csi_tensor: torch.Tensor,
    beam_weights: torch.Tensor,
):
    csi = csi_tensor.to(helper.device)
    if csi.ndim == 3:
        csi = csi.unsqueeze(0)

    weights = torch.as_tensor(beam_weights).to(helper.device)
    if weights.ndim == 2:
        weights = weights.unsqueeze(0)
    if weights.ndim != 3:
        raise ValueError(
            "beam_weights must have shape [M, N] or [num_beams, M, N]."
        )

    weights = weights.to(dtype=csi.dtype).reshape(1, weights.shape[0], -1)
    return helper.dp._generate_rss(csi, weights)[0].detach().cpu()


def effective_channels(
    helper: InferenceHelper,
    csi_tensor: torch.Tensor,
    beam_weights: torch.Tensor,
):
    csi = csi_tensor.to(helper.device)
    if csi.ndim == 3:
        csi = csi.unsqueeze(0)
    if csi.shape[0] != 1:
        raise ValueError("effective_channels expects one CSI sample.")

    weights = torch.as_tensor(beam_weights).to(helper.device)
    if weights.ndim == 2:
        weights = weights.unsqueeze(0)
    if weights.ndim != 3:
        raise ValueError(
            "beam_weights must have shape [M, N] or [num_beams, M, N]."
        )

    weights = weights.to(dtype=csi.dtype).reshape(weights.shape[0], -1)
    h_eff = torch.einsum("frt,bt->bfr", csi[0], weights)
    return h_eff[..., 0].detach().cpu()


def optimal_beam_from_csi(helper: InferenceHelper, csi_tensor: torch.Tensor):
    """Return the unit-norm wideband beam that maximizes average RSS."""
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
    return beam.reshape(
        helper.setting.dataset.M,
        helper.setting.dataset.N,
    ).detach().cpu()


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
            received = h_chunk[:, None] * symbols + noise
            detected = received / h_chunk[:, None]

            bits_hat = torch.empty_like(bits)
            bits_hat[..., 0] = (detected.real < 0).to(torch.int8)
            bits_hat[..., 1] = (detected.imag < 0).to(torch.int8)
            bit_errors += int(torch.count_nonzero(bits_hat != bits).item())
            total_bits += bits.numel()

        ber.append(bit_errors / total_bits)
    return ber


def qpsk_ber_from_gain_ratios(ebno_dbs: np.ndarray, gain_ratios):
    """Return analytic uncoded-QPSK BER using gain-ratio-scaled Eb/N0."""
    ratios = np.asarray(gain_ratios, dtype=np.float64)
    ratios = np.maximum(ratios, 1e-16)
    ber = []
    for ebno_db in ebno_dbs:
        ebno_linear = 10 ** (float(ebno_db) / 10.0)
        sample_ber = [
            0.5 * math.erfc(math.sqrt(ebno_linear * ratio)) for ratio in ratios
        ]
        ber.append(float(np.mean(sample_ber)))
    return ber


def qpsk_ber_monte_carlo_from_gain_ratios(
    ebno_dbs: np.ndarray,
    gain_ratios,
    symbols_per_sample: int | None,
    seed: int,
    total_bits: int | None = None,
    symbol_chunk_size: int = 65_536,
):
    """Simulate uncoded QPSK with Eb/N0 scaled by RSS/optimal RSS.

    When total_bits is given, that exact bit budget is distributed as evenly
    as possible over the CSI gain ratios. Otherwise, the legacy
    symbols_per_sample budget is used. Symbols are generated in chunks so a
    large bit budget does not require a correspondingly large allocation.
    """
    ratios = torch.as_tensor(gain_ratios, dtype=torch.float32).reshape(-1)
    ratios = torch.clamp(ratios, min=1e-16)
    if ratios.numel() == 0:
        return [1.0 for _ in ebno_dbs]

    if total_bits is not None:
        if total_bits < 2 or total_bits % 2 != 0:
            raise ValueError("total_bits must be a positive even number for QPSK.")
        total_symbols = total_bits // 2
    else:
        if symbols_per_sample is None or symbols_per_sample < 1:
            raise ValueError(
                "symbols_per_sample must be positive when total_bits is not set."
            )
        total_symbols = ratios.numel() * symbols_per_sample
    if symbol_chunk_size < 1:
        raise ValueError("symbol_chunk_size must be positive.")

    rng = torch.Generator(device="cpu")
    ber = []
    sqrt_two = math.sqrt(2.0)

    for ebno_db in ebno_dbs:
        rng.manual_seed(seed + int(round(float(ebno_db) * 100)))
        ebno_linear = 10 ** (float(ebno_db) / 10.0)
        effective_ebno = ebno_linear * ratios
        noise_std = torch.sqrt(1.0 / (2.0 * effective_ebno))
        bit_errors = 0
        simulated_bits = 0

        for start in range(0, total_symbols, symbol_chunk_size):
            chunk_symbols = min(symbol_chunk_size, total_symbols - start)
            sample_indices = torch.arange(
                start,
                start + chunk_symbols,
                dtype=torch.long,
            ) % ratios.numel()
            chunk_noise_std = noise_std[sample_indices]
            bits = torch.randint(
                0,
                2,
                (chunk_symbols, 2),
                generator=rng,
                dtype=torch.int8,
            )
            symbols = (
                (1.0 - 2.0 * bits[:, 0].to(torch.float32))
                + 1j * (1.0 - 2.0 * bits[:, 1].to(torch.float32))
            ) / sqrt_two
            noise = (chunk_noise_std / sqrt_two) * (
                torch.randn(symbols.shape, generator=rng)
                + 1j * torch.randn(symbols.shape, generator=rng)
            )
            received = symbols + noise

            bits_hat = torch.empty_like(bits)
            bits_hat[:, 0] = (received.real < 0).to(torch.int8)
            bits_hat[:, 1] = (received.imag < 0).to(torch.int8)
            bit_errors += int(torch.count_nonzero(bits_hat != bits).item())
            simulated_bits += bits.numel()

        ber.append(bit_errors / simulated_bits)

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
    if not rows:
        return
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
    markers = ["o", "s", "^", "D", "v", "P", "X"]
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
