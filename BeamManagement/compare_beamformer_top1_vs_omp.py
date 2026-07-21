"""
Compare BeamFormer top-1 steering-vector beamforming with top-k OMP.

The shared baseline.py module predicts the RSS over the 80 x 20 query grid,
selects the strongest predicted direction, and uses that query-grid steering
vector as the BeamFormer beamforming vector.

The OMP method uses the top-k predicted BeamFormer RSS directions as a steering
vector dictionary and approximates the optimal channel eigenvector. This assumes
the full CSI is available for the OMP target, so it is an upper-bound style
experiment for testing whether the BeamFormer spectrum can span the optimal
eigenvector.

Outputs:
  - BER comparison: Optimal, BeamFormer top-1, top-k OMP in one figure
  - Gain-loss CDF: Optimal, BeamFormer top-1, top-k OMP in one figure
  - Similarity CDF: BeamFormer top-1 and top-k OMP in one figure

Example:
    python BeamManagement/compare_beamformer_top1_vs_omp.py \
        --data_path BeamFormer/mini_demo/indoor_28g_dataset/t16x16_r2x1_test_small \
        --model_dir BeamFormer/mini_demo/saved_models \
        --max_samples 128
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR if (SCRIPT_DIR / "BeamFormer").exists() else SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import baseline


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "beamformer_top1_vs_omp_results"

np = None
plt = None
torch = None


def import_runtime_modules():
    global np
    global plt
    global torch

    try:
        baseline.import_runtime_modules()
    except SystemExit as error:
        cause = error.__cause__
        if isinstance(cause, ModuleNotFoundError):
            raise SystemExit(
                f"Missing Python package '{cause.name}'. Install BeamFormer "
                "requirements in the active environment, for example: "
                "pip install -r BeamFormer/requirements.txt"
            ) from cause
        raise

    np = baseline.np
    plt = baseline.plt
    torch = baseline.torch


def unit_column(vector, eps: float = 1e-12):
    column = torch.as_tensor(vector).reshape(-1, 1)
    norm = torch.linalg.vector_norm(column)
    return column / torch.clamp(norm, min=eps)


def normalize_dictionary_columns(dictionary, eps: float = 1e-12):
    norms = torch.linalg.vector_norm(dictionary, dim=0, keepdim=True)
    return dictionary / torch.clamp(norms, min=eps)


def complex_lstsq(dictionary, target):
    try:
        return torch.linalg.lstsq(dictionary, target).solution
    except RuntimeError:
        return torch.linalg.pinv(dictionary) @ target


def omp_approximate_column(target, dictionary, max_sparsity: int, tol: float):
    """Approximate target as a sparse linear combination of dictionary columns."""
    if dictionary.ndim != 2:
        raise ValueError("dictionary must have shape [num_antennas, num_atoms].")
    if dictionary.shape[1] < 1:
        raise ValueError("dictionary must contain at least one atom.")

    atoms = normalize_dictionary_columns(dictionary)
    target_column = unit_column(target).to(dtype=atoms.dtype, device=atoms.device)
    max_sparsity = min(max_sparsity, atoms.shape[1])

    residual = target_column.clone()
    support = []
    coeffs = None
    residual_norms = []

    for _ in range(max_sparsity):
        correlations = torch.abs(atoms.conj().transpose(0, 1) @ residual).reshape(-1)
        if support:
            correlations[torch.tensor(support, device=correlations.device)] = -1.0
        next_atom = int(torch.argmax(correlations).item())
        if next_atom in support or float(correlations[next_atom].item()) <= 0.0:
            break

        support.append(next_atom)
        active_atoms = atoms[:, support]
        coeffs = complex_lstsq(active_atoms, target_column)
        residual = target_column - active_atoms @ coeffs
        residual_norm = float(torch.linalg.vector_norm(residual).item())
        residual_norms.append(residual_norm)
        if residual_norm <= tol:
            break

    if coeffs is None:
        support = [0]
        coeffs = complex_lstsq(atoms[:, support], target_column)
        residual = target_column - atoms[:, support] @ coeffs
        residual_norms.append(float(torch.linalg.vector_norm(residual).item()))

    approximation = atoms[:, support] @ coeffs
    approximation = unit_column(approximation)
    full_coeffs = torch.zeros(
        atoms.shape[1],
        1,
        dtype=atoms.dtype,
        device=atoms.device,
    )
    full_coeffs[torch.tensor(support, device=atoms.device), :] = coeffs

    return {
        "beam_column": approximation,
        "coefficients": full_coeffs,
        "support_positions": support,
        "residual_norm": residual_norms[-1],
        "residual_norms": residual_norms,
    }


def topk_spectrum_entries(helper, spectrum, top_k: int):
    values, indices = torch.topk(spectrum.reshape(-1), k=top_k)
    rows = []
    for rank, (value, index) in enumerate(zip(values, indices), start=1):
        flat_index = int(index.item())
        phi_idx, theta_idx, phi_deg, theta_deg = baseline.index_to_angles(
            helper, flat_index
        )
        rss = max(float(value.item()), 1e-16)
        rows.append(
            {
                "rank": rank,
                "index": flat_index,
                "phi_index": phi_idx,
                "theta_index": theta_idx,
                "phi_deg": phi_deg,
                "theta_deg": theta_deg,
                "predicted_rss": float(value.item()),
                "predicted_rss_db": float(baseline.get_db(rss)),
            }
        )
    return rows


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
    markers = ["o", "s", "^"]
    for marker, (label, ber) in zip(markers, ber_curves.items()):
        ax.semilogy(ebno_dbs, ber, marker=marker, linewidth=2, label=label)
    ax.grid(True, which="both", linestyle="--", alpha=0.45)
    ax.set_xlabel("Eb/N0 [dB]")
    ax.set_ylabel("BER")
    ax.set_title("BER Comparison")
    ax.set_ylim(1e-6, 1.0)
    ax.legend()
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


def append_topk_rows(topk_rows, path: Path, entries, omp_result):
    support_to_order = {
        support_position: order
        for order, support_position in enumerate(omp_result["support_positions"], start=1)
    }
    coeffs = omp_result["coefficients"].reshape(-1).detach().cpu()

    for position, entry in enumerate(entries):
        coeff = coeffs[position]
        topk_rows.append(
            {
                "file": path.name,
                **entry,
                "selected_by_omp": position in support_to_order,
                "omp_order": support_to_order.get(position, ""),
                "omp_coeff_real": float(torch.real(coeff).item()),
                "omp_coeff_imag": float(torch.imag(coeff).item()),
                "omp_coeff_abs": float(torch.abs(coeff).item()),
            }
        )


def evaluate(helper, files, top_k: int, omp_sparsity: int, omp_tol: float, seed: int):
    reference_weights = baseline.make_reference_weights(helper)
    query_weights = helper.dp.generate_query_weights(batch_size=1).detach().cpu()

    query_num = int(query_weights.shape[1])
    top_k = min(top_k, query_num)
    omp_sparsity = min(omp_sparsity, top_k)

    optimal_label = "Optimal"
    beamformer_label = "BeamFormer top-1 steering"
    omp_label = f"BeamFormer top-{top_k} OMP"

    gain_ratio_sets = {
        optimal_label: [],
        beamformer_label: [],
        omp_label: [],
    }
    metric_rows = []
    topk_rows = []

    for path in baseline.tqdm(files, desc="[Top1-vs-OMP]"):
        csi = baseline.read_csi_file_to_torch(str(path)).unsqueeze(0)

        optimal_beam = baseline.optimal_beam_from_csi(helper, csi)
        sample_rss = baseline.simulate_csirs_rss(helper, csi, reference_weights)
        beamformer = baseline.predict_beamformer_beam(
            helper,
            sample_rss,
            reference_weights,
            query_weights,
        )

        entries = topk_spectrum_entries(helper, beamformer["spectrum"], top_k)
        topk_indices = torch.tensor(
            [entry["index"] for entry in entries],
            dtype=torch.long,
        )
        topk_atoms = query_weights[0, topk_indices].reshape(top_k, -1).transpose(0, 1)
        omp_result = omp_approximate_column(
            target=optimal_beam.reshape(-1, 1),
            dictionary=topk_atoms,
            max_sparsity=omp_sparsity,
            tol=omp_tol,
        )
        omp_beam = omp_result["beam_column"].reshape(
            helper.setting.dataset.M,
            helper.setting.dataset.N,
        )

        candidate_beams = torch.stack(
            [optimal_beam, beamformer["beam"], omp_beam],
            dim=0,
        )
        actual_rss = baseline.measure_beam_rss(helper, csi, candidate_beams)
        optimal_rss = max(float(actual_rss[0].item()), 1e-16)
        beamformer_rss = max(float(actual_rss[1].item()), 1e-16)
        omp_rss = max(float(actual_rss[2].item()), 1e-16)

        gain_ratio_sets[optimal_label].append(1.0)
        gain_ratio_sets[beamformer_label].append(beamformer_rss / optimal_rss)
        gain_ratio_sets[omp_label].append(omp_rss / optimal_rss)

        bf_phi_idx, bf_theta_idx, bf_phi_deg, bf_theta_deg = baseline.index_to_angles(
            helper,
            beamformer["index"],
        )
        selected_grid_indices = [
            entries[position]["index"]
            for position in omp_result["support_positions"]
        ]

        beamformer_loss_db = float(
            baseline.get_db(optimal_rss) - baseline.get_db(beamformer_rss)
        )
        omp_loss_db = float(baseline.get_db(optimal_rss) - baseline.get_db(omp_rss))
        beamformer_similarity = baseline.beam_similarity(
            optimal_beam,
            beamformer["beam"],
        )
        omp_similarity = baseline.beam_similarity(optimal_beam, omp_beam)

        metric_rows.append(
            {
                "file": path.name,
                "top_k": top_k,
                "omp_sparsity": omp_sparsity,
                "omp_selected_count": len(omp_result["support_positions"]),
                "omp_support_grid_indices": " ".join(str(i) for i in selected_grid_indices),
                "omp_residual_norm": omp_result["residual_norm"],
                "optimal_rss": optimal_rss,
                "beamformer_top1_rss": beamformer_rss,
                "omp_rss": omp_rss,
                "beamformer_top1_gain_ratio_to_optimal": beamformer_rss / optimal_rss,
                "omp_gain_ratio_to_optimal": omp_rss / optimal_rss,
                "optimal_gain_loss_to_optimal_db": 0.0,
                "beamformer_top1_gain_loss_to_optimal_db": beamformer_loss_db,
                "omp_gain_loss_to_optimal_db": omp_loss_db,
                "omp_gain_loss_minus_beamformer_top1_gain_loss_db": (
                    omp_loss_db - beamformer_loss_db
                ),
                "beamformer_top1_similarity_to_optimal": beamformer_similarity,
                "omp_similarity_to_optimal": omp_similarity,
                "beamformer_top1_index": beamformer["index"],
                "beamformer_top1_phi_index": bf_phi_idx,
                "beamformer_top1_theta_index": bf_theta_idx,
                "beamformer_top1_phi_deg": bf_phi_deg,
                "beamformer_top1_theta_deg": bf_theta_deg,
            }
        )
        append_topk_rows(topk_rows, path, entries, omp_result)

    gain_loss_sets = {
        optimal_label: [0.0 for _ in metric_rows],
        beamformer_label: [
            row["beamformer_top1_gain_loss_to_optimal_db"] for row in metric_rows
        ],
        omp_label: [
            row["omp_gain_loss_to_optimal_db"] for row in metric_rows
        ],
    }
    similarity_sets = {
        beamformer_label: [
            row["beamformer_top1_similarity_to_optimal"] for row in metric_rows
        ],
        omp_label: [
            row["omp_similarity_to_optimal"] for row in metric_rows
        ],
    }
    return {
        "metric_rows": metric_rows,
        "topk_rows": topk_rows,
        "gain_ratio_sets": gain_ratio_sets,
        "gain_loss_sets": gain_loss_sets,
        "similarity_sets": similarity_sets,
        "labels": {
            "optimal": optimal_label,
            "beamformer": beamformer_label,
            "omp": omp_label,
        },
        "top_k": top_k,
        "omp_sparsity": omp_sparsity,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare BeamFormer top-1 steering-vector beamforming with top-k OMP."
    )
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--generator_path", type=str, default=None)
    parser.add_argument("--estimator_path", type=str, default=None)
    parser.add_argument("--arn_model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--omp_sparsity",
        type=int,
        default=None,
        help="OMP support size. Defaults to --top_k.",
    )
    parser.add_argument("--omp_tol", type=float, default=1e-8)
    parser.add_argument("--ebno_dbs", type=str, default="-5,0,5,10,15,20")
    parser.add_argument("--ber_symbols_per_sample", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError("--top_k must be positive.")
    omp_sparsity = args.top_k if args.omp_sparsity is None else args.omp_sparsity
    if omp_sparsity < 1:
        raise ValueError("--omp_sparsity must be positive.")

    import_runtime_modules()
    torch.set_grad_enabled(False)

    data_path = (
        baseline.resolve_path(args.data_path)
        if args.data_path
        else baseline.first_existing(baseline.DEFAULT_DATA_CANDIDATES)
    )
    if data_path is None or not data_path.exists():
        candidates = "\n  ".join(str(path) for path in baseline.DEFAULT_DATA_CANDIDATES)
        raise FileNotFoundError(
            "Could not find a CSI data directory. Pass --data_path explicitly. "
            f"Default candidates were:\n  {candidates}"
        )

    model_dir = baseline.resolve_path(args.model_dir) if args.model_dir else None
    generator_path = baseline.resolve_model_file(
        args.generator_path,
        model_dir,
        ("generator_epoch_final.pth", "generator.pth"),
        "generator weights",
    )
    estimator_path = baseline.resolve_model_file(
        args.estimator_path,
        model_dir,
        ("estimator_epoch_final.pth", "estimator.pth"),
        "estimator weights",
    )
    arn_path = baseline.resolve_optional_model_file(
        args.arn_model_path,
        model_dir,
        ("model_epoch_final.pth", "arn_model.pth"),
    )

    output_dir = baseline.resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setting = baseline.build_setting(data_path, generator_path, estimator_path, arn_path)
    helper = baseline.InferenceHelper(setting)
    files = baseline.collect_files(data_path, args.max_samples)

    results = evaluate(
        helper=helper,
        files=files,
        top_k=args.top_k,
        omp_sparsity=omp_sparsity,
        omp_tol=args.omp_tol,
        seed=args.seed,
    )

    ebno_dbs = baseline.parse_ebno_dbs(args.ebno_dbs)
    ber_curves = {
        label: baseline.qpsk_ber_monte_carlo_from_gain_ratios(
            ebno_dbs,
            ratios,
            symbols_per_sample=args.ber_symbols_per_sample,
            seed=args.seed,
        )
        for label, ratios in results["gain_ratio_sets"].items()
    }

    all_gain_losses = [
        value for values in results["gain_loss_sets"].values() for value in values
    ]
    gain_loss_xlim = (
        min(-0.05, float(np.min(all_gain_losses)) - 0.05),
        max(1.0, float(np.percentile(all_gain_losses, 99)) * 1.1),
    )

    summary = {
        "num_samples": len(results["metric_rows"]),
        "data_path": str(data_path),
        "generator_path": str(generator_path),
        "estimator_path": str(estimator_path),
        "arn_model_path": str(arn_path) if arn_path is not None else None,
        "beamformer_method": (
            "argmax direction from BeamFormer predicted RSS spectrum, then use "
            "that query-grid steering vector as the beamforming vector"
        ),
        "omp_target": (
            "dominant eigenvector of mean_f H_f^H H_f for the active receiver"
        ),
        "omp_dictionary": (
            "unit-norm steering-vector columns from the BeamFormer top-k "
            "predicted RSS directions"
        ),
        "top_k": results["top_k"],
        "omp_sparsity": results["omp_sparsity"],
        "ebno_dbs": ebno_dbs.tolist(),
        "ber_mode": "monte_carlo",
        "ber_symbols_per_sample": args.ber_symbols_per_sample,
        "ber": ber_curves,
        "gain_loss_to_optimal_db": {
            label: baseline.summarize(values)
            for label, values in results["gain_loss_sets"].items()
        },
        "similarity_to_optimal": {
            label: baseline.summarize(values)
            for label, values in results["similarity_sets"].items()
        },
        "omp_gain_loss_minus_beamformer_top1_gain_loss_db": baseline.summarize(
            [
                row["omp_gain_loss_minus_beamformer_top1_gain_loss_db"]
                for row in results["metric_rows"]
            ]
        ),
    }

    write_csv(results["metric_rows"], output_dir / "per_sample_metrics.csv")
    write_csv(results["topk_rows"], output_dir / "topk_rss_and_omp_coefficients.csv")
    write_json(summary, output_dir / "summary.json")

    plot_ber(ebno_dbs, ber_curves, output_dir / "ber_comparison.png")
    plot_multi_cdf(
        results["gain_loss_sets"],
        output_dir / "gain_loss_to_optimal_cdf.png",
        "Beamforming gain loss to optimal [dB]",
        "Gain Loss to Optimal Beam",
        xlim=gain_loss_xlim,
    )
    plot_multi_cdf(
        results["similarity_sets"],
        output_dir / "similarity_to_optimal_cdf.png",
        "Normalized beam-vector similarity",
        "Similarity to Optimal Beam",
        xlim=(-0.02, 1.02),
    )

    labels = results["labels"]
    print(f"Wrote metrics CSV:      {output_dir / 'per_sample_metrics.csv'}")
    print(f"Wrote top-k CSV:        {output_dir / 'topk_rss_and_omp_coefficients.csv'}")
    print(f"Wrote summary JSON:     {output_dir / 'summary.json'}")
    print(f"Wrote BER figure:       {output_dir / 'ber_comparison.png'}")
    print(f"Wrote gain-loss figure: {output_dir / 'gain_loss_to_optimal_cdf.png'}")
    print(f"Wrote similarity figure:{output_dir / 'similarity_to_optimal_cdf.png'}")
    print(
        "Median BeamFormer loss [dB]: "
        f"{summary['gain_loss_to_optimal_db'][labels['beamformer']]['median']:.4f}"
    )
    print(
        "Median OMP loss [dB]:        "
        f"{summary['gain_loss_to_optimal_db'][labels['omp']]['median']:.4f}"
    )
    print(
        "Mean BeamFormer similarity:  "
        f"{summary['similarity_to_optimal'][labels['beamformer']]['mean']:.4f}"
    )
    print(
        "Mean OMP similarity:         "
        f"{summary['similarity_to_optimal'][labels['omp']]['mean']:.4f}"
    )


if __name__ == "__main__":
    main()
