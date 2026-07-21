"""Compare OMP eigenvector approximations for several steering-vector counts.

For every CSI sample, BeamFormer reconstructs the 80 x 20 RSS spectrum once.
The strongest K query-grid directions form the OMP dictionary for
K in {1, 3, 5, 10, 20, 50}. OMP approximates the dominant channel
eigenvector using up to K atoms. Full CSI is used only to construct the OMP
target, so this is an upper-bound experiment rather than a deployable method.

The script creates three figures. Each figure contains Optimal and all OMP-K
methods:

  - BER versus Eb/N0
  - beamforming gain-loss CDF
  - similarity-to-optimal CDF

Example:
    python BeamManagement/compare_omp_steering_vector_counts.py \
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
import compare_beamformer_top1_vs_omp as common
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "omp_steering_count_comparison_results"

np = None
plt = None
torch = None


def import_runtime_modules():
    global np
    global plt
    global torch

    baseline.import_runtime_modules()
    common.np = baseline.np
    common.plt = baseline.plt
    common.torch = baseline.torch
    np = baseline.np
    plt = baseline.plt
    torch = baseline.torch


def parse_top_ks(value: str) -> list[int]:
    top_ks = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        top_k = int(token)
        if top_k < 1:
            raise ValueError("Every --top_ks value must be positive.")
        if top_k not in top_ks:
            top_ks.append(top_k)
    if not top_ks:
        raise ValueError("--top_ks must contain at least one positive integer.")
    return top_ks


def write_csv(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def plot_ber(ebno_dbs, ber_curves, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.4, 5.7), dpi=180)
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    for marker, (label, ber) in zip(markers, ber_curves.items()):
        is_optimal = label == "Optimal"
        ax.semilogy(
            ebno_dbs,
            ber,
            marker=marker,
            markersize=5.5,
            linewidth=2.6 if is_optimal else 1.8,
            color="black" if is_optimal else None,
            label=label,
        )

    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_xlabel("Eb/N0 [dB]")
    ax.set_ylabel("BER")
    ax.set_title("BER: OMP Steering-Vector Count Comparison")
    ax.set_ylim(bottom=1e-6, top=1.0)
    ax.legend(ncol=2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_multi_cdf(series, output_path: Path, xlabel: str, title: str, xlim):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.4, 5.7), dpi=180)

    for label, raw_values in series.items():
        values = np.sort(np.asarray(raw_values, dtype=np.float64))
        if values.size == 0:
            continue
        cdf = np.arange(1, values.size + 1, dtype=np.float64) / values.size
        x = np.r_[values[0], values]
        y = np.r_[0.0, cdf]
        is_optimal = label == "Optimal"
        ax.step(
            x,
            y,
            where="post",
            linewidth=2.6 if is_optimal else 1.8,
            color="black" if is_optimal else None,
            label=label,
        )

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Empirical CDF")
    ax.set_title(title)
    ax.set_xlim(*xlim)
    ax.set_ylim(0.0, 1.0)
    ax.legend(ncol=2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def evaluate(helper, files, requested_top_ks: list[int], omp_tol: float):
    reference_weights = baseline.make_reference_weights(helper)
    query_weights = helper.dp.generate_query_weights(batch_size=1).detach().cpu()
    query_count = int(query_weights.shape[1])
    top_ks = [min(top_k, query_count) for top_k in requested_top_ks]
    top_ks = list(dict.fromkeys(top_ks))
    max_top_k = max(top_ks)

    optimal_label = "Optimal"
    labels = {top_k: f"OMP top-{top_k}" for top_k in top_ks}
    gain_ratio_sets = {optimal_label: []}
    gain_loss_sets = {optimal_label: []}
    similarity_sets = {optimal_label: []}
    for top_k in top_ks:
        gain_ratio_sets[labels[top_k]] = []
        gain_loss_sets[labels[top_k]] = []
        similarity_sets[labels[top_k]] = []

    metric_rows = []
    spectrum_rows = []

    for path in baseline.tqdm(files, desc="[OMP steering counts]"):
        csi = baseline.read_csi_file_to_torch(str(path)).unsqueeze(0)
        optimal_beam = baseline.optimal_beam_from_csi(helper, csi)
        sample_rss = baseline.simulate_csirs_rss(helper, csi, reference_weights)
        beamformer = baseline.predict_beamformer_beam(
            helper,
            sample_rss,
            reference_weights,
            query_weights,
        )

        entries = common.topk_spectrum_entries(
            helper,
            beamformer["spectrum"],
            max_top_k,
        )
        max_top_indices = torch.tensor(
            [entry["index"] for entry in entries],
            dtype=torch.long,
        )
        max_top_atoms = query_weights[0, max_top_indices].reshape(
            max_top_k,
            -1,
        ).transpose(0, 1)

        omp_results = {}
        omp_beams = []
        for top_k in top_ks:
            omp_result = common.omp_approximate_column(
                target=optimal_beam.reshape(-1, 1),
                dictionary=max_top_atoms[:, :top_k],
                max_sparsity=top_k,
                tol=omp_tol,
            )
            omp_results[top_k] = omp_result
            omp_beams.append(
                omp_result["beam_column"].reshape(
                    helper.setting.dataset.M,
                    helper.setting.dataset.N,
                )
            )

        candidate_beams = torch.stack([optimal_beam, *omp_beams], dim=0)
        actual_rss = baseline.measure_beam_rss(helper, csi, candidate_beams)
        optimal_rss = max(float(actual_rss[0].item()), 1e-16)

        gain_ratio_sets[optimal_label].append(1.0)
        gain_loss_sets[optimal_label].append(0.0)
        similarity_sets[optimal_label].append(1.0)

        for position, (top_k, omp_beam) in enumerate(zip(top_ks, omp_beams), start=1):
            label = labels[top_k]
            omp_result = omp_results[top_k]
            omp_rss = max(float(actual_rss[position].item()), 1e-16)
            gain_ratio = omp_rss / optimal_rss
            gain_loss_db = float(
                baseline.get_db(optimal_rss) - baseline.get_db(omp_rss)
            )
            similarity = baseline.beam_similarity(optimal_beam, omp_beam)
            support_grid_indices = [
                entries[support_position]["index"]
                for support_position in omp_result["support_positions"]
            ]

            gain_ratio_sets[label].append(gain_ratio)
            gain_loss_sets[label].append(gain_loss_db)
            similarity_sets[label].append(similarity)
            metric_rows.append(
                {
                    "file": path.name,
                    "top_k": top_k,
                    "omp_max_sparsity": top_k,
                    "omp_selected_count": len(omp_result["support_positions"]),
                    "omp_support_grid_indices": " ".join(
                        str(index) for index in support_grid_indices
                    ),
                    "omp_residual_norm": omp_result["residual_norm"],
                    "optimal_rss": optimal_rss,
                    "omp_rss": omp_rss,
                    "gain_ratio_to_optimal": gain_ratio,
                    "gain_loss_to_optimal_db": gain_loss_db,
                    "similarity_to_optimal": similarity,
                }
            )

        for entry in entries:
            spectrum_rows.append({"file": path.name, **entry})

    return {
        "top_ks": top_ks,
        "labels": labels,
        "metric_rows": metric_rows,
        "spectrum_rows": spectrum_rows,
        "gain_ratio_sets": gain_ratio_sets,
        "gain_loss_sets": gain_loss_sets,
        "similarity_sets": similarity_sets,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare optimal beamforming with OMP using several BeamFormer "
            "top-K steering-vector dictionaries."
        )
    )
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--generator_path", type=str, default=None)
    parser.add_argument("--estimator_path", type=str, default=None)
    parser.add_argument("--arn_model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--top_ks", type=str, default="1,3,5,10,20,50")
    parser.add_argument("--omp_tol", type=float, default=1e-8)
    parser.add_argument("--ebno_dbs", type=str, default="-5,0,5,10,15,20")
    parser.add_argument(
        "--ber_bits_per_snr",
        type=int,
        default=baseline.DEFAULT_MONTE_CARLO_BITS,
        help="Total QPSK bits simulated for each method at each Eb/N0 point.",
    )
    parser.add_argument(
        "--ber_symbols_per_sample",
        type=int,
        default=None,
        help=(
            "Legacy per-CSI symbol budget. When set, it overrides "
            "--ber_bits_per_snr."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    requested_top_ks = parse_top_ks(args.top_ks)
    if args.ber_bits_per_snr < 2 or args.ber_bits_per_snr % 2 != 0:
        raise ValueError("--ber_bits_per_snr must be a positive even number.")
    if args.ber_symbols_per_sample is not None and args.ber_symbols_per_sample < 1:
        raise ValueError("--ber_symbols_per_sample must be positive when set.")
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
    results = evaluate(helper, files, requested_top_ks, args.omp_tol)

    ebno_dbs = baseline.parse_ebno_dbs(args.ebno_dbs)
    ber_curves = {
        label: baseline.qpsk_ber_monte_carlo_from_gain_ratios(
            ebno_dbs,
            gain_ratios,
            symbols_per_sample=args.ber_symbols_per_sample,
            seed=args.seed,
            total_bits=(
                None
                if args.ber_symbols_per_sample is not None
                else args.ber_bits_per_snr
            ),
        )
        for label, gain_ratios in results["gain_ratio_sets"].items()
    }

    all_gain_losses = [
        value
        for values in results["gain_loss_sets"].values()
        for value in values
    ]
    gain_loss_xlim = (
        min(-0.05, float(np.min(all_gain_losses)) - 0.05),
        max(1.0, float(np.percentile(all_gain_losses, 99)) * 1.1),
    )

    summary = {
        "num_samples": len(files),
        "data_path": str(data_path),
        "generator_path": str(generator_path),
        "estimator_path": str(estimator_path),
        "arn_model_path": str(arn_path) if arn_path is not None else None,
        "beamformer_spectrum_size": int(
            helper.setting.assumption.angle_spectrum_length
        ),
        "top_ks": results["top_ks"],
        "omp_max_sparsity": "equal to top_k for each curve",
        "omp_target": (
            "dominant eigenvector of mean_f H_f^H H_f for the active receiver"
        ),
        "omp_dictionary": (
            "unit-norm steering-vector columns from the BeamFormer top-K "
            "predicted RSS directions"
        ),
        "ebno_dbs": ebno_dbs.tolist(),
        "ber_mode": "QPSK Monte Carlo from gain ratios to optimal",
        "ber_bits_per_snr": (
            2 * len(files) * args.ber_symbols_per_sample
            if args.ber_symbols_per_sample is not None
            else args.ber_bits_per_snr
        ),
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
    }

    metrics_path = output_dir / "per_sample_omp_count_metrics.csv"
    spectrum_path = output_dir / "top50_rss_directions.csv"
    summary_path = output_dir / "summary.json"
    ber_path = output_dir / "ber_omp_count_comparison.png"
    gain_loss_path = output_dir / "gain_loss_omp_count_cdf.png"
    similarity_path = output_dir / "similarity_omp_count_cdf.png"

    write_csv(results["metric_rows"], metrics_path)
    write_csv(results["spectrum_rows"], spectrum_path)
    write_json(summary, summary_path)
    plot_ber(ebno_dbs, ber_curves, ber_path)
    plot_multi_cdf(
        results["gain_loss_sets"],
        gain_loss_path,
        "Beamforming gain loss to optimal [dB]",
        "Gain Loss: OMP Steering-Vector Count Comparison",
        gain_loss_xlim,
    )
    plot_multi_cdf(
        results["similarity_sets"],
        similarity_path,
        "Normalized beam-vector similarity",
        "Similarity: OMP Steering-Vector Count Comparison",
        (-0.02, 1.02),
    )

    print(f"Wrote metrics CSV:      {metrics_path}")
    print(f"Wrote top-50 RSS CSV:   {spectrum_path}")
    print(f"Wrote summary JSON:     {summary_path}")
    print(f"Wrote BER figure:       {ber_path}")
    print(f"Wrote gain-loss figure: {gain_loss_path}")
    print(f"Wrote similarity figure:{similarity_path}")
    for top_k in results["top_ks"]:
        label = results["labels"][top_k]
        median_loss = summary["gain_loss_to_optimal_db"][label]["median"]
        mean_similarity = summary["similarity_to_optimal"][label]["mean"]
        print(
            f"K={top_k:>2}: median loss={median_loss:.4f} dB, "
            f"mean similarity={mean_similarity:.4f}"
        )


if __name__ == "__main__":
    main()
