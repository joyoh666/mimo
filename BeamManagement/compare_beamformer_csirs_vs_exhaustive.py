"""Compare BeamFormer CSI-RS selection against a same-overhead beam sweep.

Shared model loading, channel metrics, baseline beams, BER simulation, and
plotting live in baseline.py. This file contains only this experiment's
comparison loop and command-line entry point.

Example:
    python BeamManagement/compare_beamformer_csirs_vs_exhaustive.py \\
        --data_path BeamFormer/csi-dataset/homeoffice-communication-28G-csi/t16x16_r2x1_test_small \\
        --model_dir BeamFormer/saved_models/co_train \\
        --max_samples 128
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import baseline


DEFAULT_OUTPUT_DIR = baseline.PROJECT_ROOT / "comparison_results"


def evaluate(
    helper,
    files,
    include_best_reference: bool,
    sweep_size: int,
    sweep_mode: str,
    seed: int,
):
    reference_weights = baseline.make_reference_weights(helper)
    query_weights = helper.dp.generate_query_weights(batch_size=1).detach().cpu()
    candidate_indices, sweep_label = baseline.make_query_grid_candidate_indices(
        helper,
        sweep_size=sweep_size,
        sweep_mode=sweep_mode,
        seed=seed,
    )

    rows = []
    gain_ratio_sets = {
        "Optimal": [],
        sweep_label: [],
        "BeamFormer CSI-RS": [],
    }
    if include_best_reference:
        gain_ratio_sets["Best measured CSI-RS ref"] = []

    for path in baseline.tqdm(files, desc="[Compare]"):
        csi = baseline.read_csi_file_to_torch(str(path)).unsqueeze(0)
        optimal_beam = baseline.optimal_beam_from_csi(helper, csi)
        sample_rss = baseline.simulate_csirs_rss(helper, csi, reference_weights)
        beamformer = baseline.predict_beamformer_beam(
            helper,
            sample_rss,
            reference_weights,
            query_weights,
        )
        sweep = baseline.select_sweep_beam(
            helper,
            csi,
            query_weights,
            candidate_indices,
        )

        candidate_beams = [optimal_beam, sweep["beam"], beamformer["beam"]]
        best_reference_index = int(baseline.torch.argmax(sample_rss).item())
        if include_best_reference:
            candidate_beams.append(reference_weights[0, best_reference_index])

        candidate_beams = baseline.torch.stack(candidate_beams, dim=0)
        actual_rss = baseline.measure_beam_rss(helper, csi, candidate_beams)
        optimal_rss = max(float(actual_rss[0].item()), 1e-16)
        sweep_rss = max(float(actual_rss[1].item()), 1e-16)
        beamformer_rss = max(float(actual_rss[2].item()), 1e-16)

        gain_ratio_sets["Optimal"].append(1.0)
        gain_ratio_sets[sweep_label].append(sweep_rss / optimal_rss)
        gain_ratio_sets["BeamFormer CSI-RS"].append(beamformer_rss / optimal_rss)

        bf_phi_idx, bf_theta_idx, bf_phi_deg, bf_theta_deg = baseline.index_to_angles(
            helper,
            beamformer["index"],
        )
        sweep_phi_idx, sweep_theta_idx, sweep_phi_deg, sweep_theta_deg = (
            baseline.index_to_angles(helper, sweep["index"])
        )
        sweep_similarity = baseline.beam_similarity(optimal_beam, sweep["beam"])
        beamformer_similarity = baseline.beam_similarity(
            optimal_beam,
            beamformer["beam"],
        )
        sweep_gain_loss_db = float(
            baseline.get_db(optimal_rss) - baseline.get_db(sweep_rss)
        )
        beamformer_gain_loss_db = float(
            baseline.get_db(optimal_rss) - baseline.get_db(beamformer_rss)
        )

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
            "beamformer_vs_sweep_phi_error_deg": baseline.wrapped_phi_error_deg(
                bf_phi_deg,
                sweep_phi_deg,
            ),
            "beamformer_vs_sweep_theta_error_deg": abs(
                bf_theta_deg - sweep_theta_deg
            ),
            "optimal_rss": optimal_rss,
            "beamformer_rss": beamformer_rss,
            "sweep_rss": sweep_rss,
            "sweep_gain_loss_to_optimal_db": sweep_gain_loss_db,
            "beamformer_gain_loss_to_optimal_db": beamformer_gain_loss_db,
            "best_reference_index": best_reference_index,
        }
        if include_best_reference:
            ref_rss = max(float(actual_rss[3].item()), 1e-16)
            ref_similarity = baseline.beam_similarity(
                optimal_beam,
                reference_weights[0, best_reference_index],
            )
            gain_ratio_sets["Best measured CSI-RS ref"].append(ref_rss / optimal_rss)
            row["best_reference_rss"] = ref_rss
            row["best_reference_similarity_to_optimal"] = ref_similarity
            row["best_reference_loss_to_optimal_db"] = float(
                baseline.get_db(optimal_rss) - baseline.get_db(ref_rss)
            )
        rows.append(row)

    return rows, gain_ratio_sets, sweep_label, len(candidate_indices)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare BeamFormer CSI-RS selection with same-overhead "
            "query-grid sweeping."
        )
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
        help=(
            "BER estimation method. Monte Carlo simulates QPSK symbols using "
            "gain-ratio-scaled Eb/N0."
        ),
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

    baseline.import_runtime_modules()
    baseline.torch.set_grad_enabled(False)

    data_path = (
        baseline.resolve_path(args.data_path)
        if args.data_path
        else baseline.first_existing(baseline.DEFAULT_DATA_CANDIDATES)
    )
    if data_path is None or not data_path.exists():
        candidates = "\n  ".join(
            str(path) for path in baseline.DEFAULT_DATA_CANDIDATES
        )
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
    setting = baseline.build_setting(
        data_path,
        generator_path,
        estimator_path,
        arn_path,
    )
    helper = baseline.InferenceHelper(setting)
    files = baseline.collect_files(data_path, args.max_samples)

    rows, gain_ratio_sets, sweep_label, actual_sweep_size = evaluate(
        helper,
        files,
        args.include_best_reference,
        sweep_size=args.sweep_size,
        sweep_mode=args.sweep_mode,
        seed=args.seed,
    )

    ebno_dbs = baseline.parse_ebno_dbs(args.ebno_dbs)
    if args.ber_mode == "monte_carlo":
        ber_curves = {
            label: baseline.qpsk_ber_monte_carlo_from_gain_ratios(
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
            label: baseline.qpsk_ber_from_gain_ratios(ebno_dbs, ratios)
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
        min(-0.05, float(baseline.np.min(all_gain_losses)) - 0.05),
        max(1.0, float(baseline.np.percentile(all_gain_losses, 99)) * 1.1),
    )
    summary = {
        "num_samples": len(rows),
        "data_path": str(data_path),
        "generator_path": str(generator_path),
        "estimator_path": str(estimator_path),
        "arn_model_path": str(arn_path) if arn_path is not None else None,
        "optimal_beam_definition": (
            "dominant eigenvector of mean_f H_f^H H_f for the active receiver"
        ),
        "sweep_baseline": {
            "label": sweep_label,
            "mode": args.sweep_mode,
            "requested_sweep_size": args.sweep_size,
            "actual_sweep_size": actual_sweep_size,
            "query_grid": "80 x 20 BeamFormer query beam grid",
        },
        "beamformer_sweep_match_rate": float(
            baseline.np.mean([row["beamformer_matches_sweep"] for row in rows])
        ),
        "similarity_to_optimal": {
            label: baseline.summarize(values)
            for label, values in similarity_sets.items()
        },
        "gain_loss_to_optimal_db": {
            label: baseline.summarize(values)
            for label, values in gain_loss_sets.items()
        },
        "ebno_dbs": ebno_dbs.tolist(),
        "ber_mode": args.ber_mode,
        "ber_type": ber_type,
        "ber_symbols_per_sample": args.ber_symbols_per_channel,
        "ber": ber_curves,
    }

    baseline.write_csv(rows, output_dir / "per_sample_metrics.csv")
    baseline.write_json(summary, output_dir / "summary.json")
    baseline.plot_ber(ebno_dbs, ber_curves, output_dir / "ber_comparison.png")
    baseline.plot_cdf(
        similarity_sets[f"Optimal vs {sweep_label}"],
        output_dir / "similarity_optimal_vs_sweep_cdf.png",
        "Normalized beam-vector similarity",
        "Optimal vs Sparse-Sweep Beam Similarity",
        xlim=(-0.02, 1.02),
    )
    baseline.plot_cdf(
        similarity_sets["Optimal vs BeamFormer"],
        output_dir / "similarity_optimal_vs_beamformer_cdf.png",
        "Normalized beam-vector similarity",
        "Optimal vs BeamFormer Beam Similarity",
        xlim=(-0.02, 1.02),
    )
    baseline.plot_multi_cdf(
        similarity_sets,
        output_dir / "similarity_to_optimal_cdf.png",
        "Normalized beam-vector similarity",
        "Similarity to Optimal Beam",
        xlim=(-0.02, 1.02),
    )
    baseline.plot_cdf(
        gain_loss_sets[f"Optimal vs {sweep_label}"],
        output_dir / "gain_loss_optimal_vs_sweep_cdf.png",
        "Beamforming gain loss to optimal [dB]",
        "Sparse-Sweep Loss to Optimal Beam",
        xlim=gain_loss_xlim,
    )
    baseline.plot_cdf(
        gain_loss_sets["Optimal vs BeamFormer"],
        output_dir / "gain_loss_optimal_vs_beamformer_cdf.png",
        "Beamforming gain loss to optimal [dB]",
        "BeamFormer Loss to Optimal Beam",
        xlim=gain_loss_xlim,
    )
    baseline.plot_multi_cdf(
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

