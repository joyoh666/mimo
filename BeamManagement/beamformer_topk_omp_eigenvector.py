"""
Approximate the channel eigenvector from BeamFormer top-k RSS directions.

This script is intentionally additive: it does not modify the BeamFormer
package or the existing comparison scripts. It reconstructs the 1,600-point
BeamFormer RSS spectrum, prints/saves the top-k directions, builds a steering
vector dictionary from those directions, and uses OMP to approximate the
optimal channel eigenvector. The resulting column vector is used as a
beamforming vector and compared against:

  - the unconstrained optimal eigenvector
  - a 64-beam uniform query-grid sweep baseline

Example:
    python BeamManagement/beamformer_topk_omp_eigenvector.py \
        --data_path BeamFormer/mini_demo/indoor_28g_dataset/t16x16_r2x1_test_small \
        --model_dir BeamFormer/mini_demo/saved_models \
        --max_samples 32
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


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "topk_omp_results"

np = None
torch = None


def import_runtime_modules():
    baseline.import_runtime_modules()
    global np
    global torch
    np = baseline.np
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
    residual_norms = []
    coeffs = None

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
        rows.append(
            {
                "rank": rank,
                "index": flat_index,
                "phi_index": phi_idx,
                "theta_index": theta_idx,
                "phi_deg": phi_deg,
                "theta_deg": theta_deg,
                "predicted_rss": float(value.item()),
                "predicted_rss_db": float(baseline.get_db(max(float(value.item()), 1e-16))),
            }
        )
    return rows


def print_topk_entries(file_name: str, entries):
    print(f"\nTop-{len(entries)} BeamFormer RSS directions for {file_name}")
    print("rank,index,phi_deg,theta_deg,predicted_rss,predicted_rss_db")
    for item in entries:
        print(
            f"{item['rank']},{item['index']},"
            f"{item['phi_deg']:.3f},{item['theta_deg']:.3f},"
            f"{item['predicted_rss']:.6e},{item['predicted_rss_db']:.3f}"
        )


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


def append_topk_rows(topk_rows, path: Path, entries, omp_result):
    support_to_order = {
        support_position: order
        for order, support_position in enumerate(omp_result["support_positions"], start=1)
    }
    coeffs = omp_result["coefficients"].reshape(-1).detach().cpu()

    for position, entry in enumerate(entries):
        coeff = coeffs[position]
        row = {
            "file": path.name,
            **entry,
            "selected_by_omp": position in support_to_order,
            "omp_order": support_to_order.get(position, ""),
            "omp_coeff_real": float(torch.real(coeff).item()),
            "omp_coeff_imag": float(torch.imag(coeff).item()),
            "omp_coeff_abs": float(torch.abs(coeff).item()),
        }
        topk_rows.append(row)


def qpsk_ber_curves(ebno_dbs, gain_ratio_sets, symbols_per_sample: int, seed: int):
    return {
        label: baseline.qpsk_ber_monte_carlo_from_gain_ratios(
            ebno_dbs,
            ratios,
            symbols_per_sample=symbols_per_sample,
            seed=seed,
        )
        for label, ratios in gain_ratio_sets.items()
    }


def save_vectors_npz(output_path: Path, vector_payload):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        files=np.array(vector_payload["files"]),
        topk_indices=np.asarray(vector_payload["topk_indices"], dtype=np.int64),
        topk_phi_deg=np.asarray(vector_payload["topk_phi_deg"], dtype=np.float64),
        topk_theta_deg=np.asarray(vector_payload["topk_theta_deg"], dtype=np.float64),
        topk_predicted_rss=np.asarray(
            vector_payload["topk_predicted_rss"], dtype=np.float64
        ),
        omp_support_positions=np.asarray(
            vector_payload["omp_support_positions"], dtype=np.int64
        ),
        omp_coefficients=np.stack(vector_payload["omp_coefficients"]),
        omp_beam_columns=np.stack(vector_payload["omp_beam_columns"]),
        optimal_beam_columns=np.stack(vector_payload["optimal_beam_columns"]),
        sweep_beam_columns=np.stack(vector_payload["sweep_beam_columns"]),
        beamformer_top1_columns=np.stack(vector_payload["beamformer_top1_columns"]),
    )


def evaluate(
    helper,
    files,
    top_k: int,
    omp_sparsity: int,
    omp_tol: float,
    sweep_size: int,
    sweep_mode: str,
    seed: int,
    print_topk_samples: int,
    include_top1_curve: bool,
):
    reference_weights = baseline.make_reference_weights(helper)
    query_weights = helper.dp.generate_query_weights(batch_size=1).detach().cpu()
    candidate_indices, sweep_label = baseline.make_query_grid_candidate_indices(
        helper,
        sweep_size=sweep_size,
        sweep_mode=sweep_mode,
        seed=seed,
    )

    query_num = int(query_weights.shape[1])
    top_k = min(top_k, query_num)
    omp_sparsity = min(omp_sparsity, top_k)

    omp_label = f"BeamFormer top-{top_k} OMP"
    gain_ratio_sets = {
        "Optimal": [],
        sweep_label: [],
        omp_label: [],
    }
    if include_top1_curve:
        gain_ratio_sets["BeamFormer top-1 query beam"] = []

    metric_rows = []
    topk_rows = []
    vector_payload = {
        "files": [],
        "topk_indices": [],
        "topk_phi_deg": [],
        "topk_theta_deg": [],
        "topk_predicted_rss": [],
        "omp_support_positions": [],
        "omp_coefficients": [],
        "omp_beam_columns": [],
        "optimal_beam_columns": [],
        "sweep_beam_columns": [],
        "beamformer_top1_columns": [],
    }

    for sample_idx, path in enumerate(baseline.tqdm(files, desc="[TopK-OMP]")):
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
            [optimal_beam, sweep["beam"], beamformer["beam"], omp_beam],
            dim=0,
        )
        actual_rss = baseline.measure_beam_rss(helper, csi, candidate_beams)
        optimal_rss = max(float(actual_rss[0].item()), 1e-16)
        sweep_rss = max(float(actual_rss[1].item()), 1e-16)
        top1_rss = max(float(actual_rss[2].item()), 1e-16)
        omp_rss = max(float(actual_rss[3].item()), 1e-16)

        gain_ratio_sets["Optimal"].append(1.0)
        gain_ratio_sets[sweep_label].append(sweep_rss / optimal_rss)
        gain_ratio_sets[omp_label].append(omp_rss / optimal_rss)
        if include_top1_curve:
            gain_ratio_sets["BeamFormer top-1 query beam"].append(top1_rss / optimal_rss)

        top1_phi_idx, top1_theta_idx, top1_phi_deg, top1_theta_deg = (
            baseline.index_to_angles(helper, beamformer["index"])
        )
        sweep_phi_idx, sweep_theta_idx, sweep_phi_deg, sweep_theta_deg = (
            baseline.index_to_angles(helper, sweep["index"])
        )
        selected_grid_indices = [
            entries[position]["index"]
            for position in omp_result["support_positions"]
        ]

        sweep_loss_db = float(baseline.get_db(optimal_rss) - baseline.get_db(sweep_rss))
        top1_loss_db = float(baseline.get_db(optimal_rss) - baseline.get_db(top1_rss))
        omp_loss_db = float(baseline.get_db(optimal_rss) - baseline.get_db(omp_rss))

        metric_rows.append(
            {
                "file": path.name,
                "top_k": top_k,
                "omp_sparsity": omp_sparsity,
                "omp_selected_count": len(omp_result["support_positions"]),
                "omp_support_grid_indices": " ".join(str(i) for i in selected_grid_indices),
                "omp_residual_norm": omp_result["residual_norm"],
                "optimal_rss": optimal_rss,
                "sweep_rss": sweep_rss,
                "beamformer_top1_rss": top1_rss,
                "omp_rss": omp_rss,
                "sweep_gain_ratio_to_optimal": sweep_rss / optimal_rss,
                "beamformer_top1_gain_ratio_to_optimal": top1_rss / optimal_rss,
                "omp_gain_ratio_to_optimal": omp_rss / optimal_rss,
                "sweep_gain_loss_to_optimal_db": sweep_loss_db,
                "beamformer_top1_gain_loss_to_optimal_db": top1_loss_db,
                "omp_gain_loss_to_optimal_db": omp_loss_db,
                "omp_gain_loss_minus_sweep_gain_loss_db": omp_loss_db - sweep_loss_db,
                "sweep_similarity_to_optimal": baseline.beam_similarity(
                    optimal_beam,
                    sweep["beam"],
                ),
                "beamformer_top1_similarity_to_optimal": baseline.beam_similarity(
                    optimal_beam,
                    beamformer["beam"],
                ),
                "omp_similarity_to_optimal": baseline.beam_similarity(
                    optimal_beam,
                    omp_beam,
                ),
                "beamformer_top1_index": beamformer["index"],
                "beamformer_top1_phi_index": top1_phi_idx,
                "beamformer_top1_theta_index": top1_theta_idx,
                "beamformer_top1_phi_deg": top1_phi_deg,
                "beamformer_top1_theta_deg": top1_theta_deg,
                "sweep_index": sweep["index"],
                "sweep_phi_index": sweep_phi_idx,
                "sweep_theta_index": sweep_theta_idx,
                "sweep_phi_deg": sweep_phi_deg,
                "sweep_theta_deg": sweep_theta_deg,
            }
        )

        append_topk_rows(topk_rows, path, entries, omp_result)

        if sample_idx < print_topk_samples:
            print_topk_entries(path.name, entries)

        padded_support = [-1] * top_k
        for pos, support_position in enumerate(omp_result["support_positions"]):
            padded_support[pos] = support_position

        vector_payload["files"].append(path.name)
        vector_payload["topk_indices"].append([entry["index"] for entry in entries])
        vector_payload["topk_phi_deg"].append([entry["phi_deg"] for entry in entries])
        vector_payload["topk_theta_deg"].append([entry["theta_deg"] for entry in entries])
        vector_payload["topk_predicted_rss"].append(
            [entry["predicted_rss"] for entry in entries]
        )
        vector_payload["omp_support_positions"].append(padded_support)
        vector_payload["omp_coefficients"].append(
            omp_result["coefficients"].reshape(-1).detach().cpu().numpy()
        )
        vector_payload["omp_beam_columns"].append(
            omp_result["beam_column"].detach().cpu().numpy()
        )
        vector_payload["optimal_beam_columns"].append(
            unit_column(optimal_beam).detach().cpu().numpy()
        )
        vector_payload["sweep_beam_columns"].append(
            unit_column(sweep["beam"]).detach().cpu().numpy()
        )
        vector_payload["beamformer_top1_columns"].append(
            unit_column(beamformer["beam"]).detach().cpu().numpy()
        )

    return metric_rows, topk_rows, gain_ratio_sets, vector_payload, sweep_label, omp_label


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Use BeamFormer top-k RSS directions as an OMP dictionary for "
            "approximating the dominant channel eigenvector."
        )
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
    parser.add_argument(
        "--sweep_size",
        type=int,
        default=64,
        help="Number of measured beams in the query-grid sweep baseline.",
    )
    parser.add_argument(
        "--sweep_mode",
        type=str,
        default="uniform_query_grid",
        choices=["uniform_query_grid", "random_query_grid", "full_query_grid"],
    )
    parser.add_argument("--ebno_dbs", type=str, default="-5,0,5,10,15,20")
    parser.add_argument("--ber_symbols_per_sample", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--print_topk_samples",
        type=int,
        default=3,
        help="Print top-k RSS tables for the first N samples; all samples are saved to CSV.",
    )
    parser.add_argument(
        "--include_top1_curve",
        action="store_true",
        help="Also include the BeamFormer top-1 query beam in the BER plot.",
    )
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

    (
        metric_rows,
        topk_rows,
        gain_ratio_sets,
        vector_payload,
        sweep_label,
        omp_label,
    ) = evaluate(
        helper=helper,
        files=files,
        top_k=args.top_k,
        omp_sparsity=omp_sparsity,
        omp_tol=args.omp_tol,
        sweep_size=args.sweep_size,
        sweep_mode=args.sweep_mode,
        seed=args.seed,
        print_topk_samples=args.print_topk_samples,
        include_top1_curve=args.include_top1_curve,
    )

    ebno_dbs = baseline.parse_ebno_dbs(args.ebno_dbs)
    ber_curves = qpsk_ber_curves(
        ebno_dbs,
        gain_ratio_sets,
        symbols_per_sample=args.ber_symbols_per_sample,
        seed=args.seed,
    )

    gain_loss_sets = {
        f"Optimal vs {sweep_label}": [
            row["sweep_gain_loss_to_optimal_db"] for row in metric_rows
        ],
        f"Optimal vs {omp_label}": [
            row["omp_gain_loss_to_optimal_db"] for row in metric_rows
        ],
    }
    similarity_sets = {
        f"Optimal vs {sweep_label}": [
            row["sweep_similarity_to_optimal"] for row in metric_rows
        ],
        f"Optimal vs {omp_label}": [
            row["omp_similarity_to_optimal"] for row in metric_rows
        ],
    }

    summary = {
        "num_samples": len(metric_rows),
        "data_path": str(data_path),
        "generator_path": str(generator_path),
        "estimator_path": str(estimator_path),
        "arn_model_path": str(arn_path) if arn_path is not None else None,
        "beamformer_spectrum_size": helper.setting.assumption.angle_spectrum_length,
        "top_k": args.top_k,
        "omp_sparsity": min(omp_sparsity, args.top_k),
        "omp_target": (
            "dominant eigenvector of mean_f H_f^H H_f for the active receiver"
        ),
        "omp_dictionary": (
            "unit-norm steering-vector columns from the BeamFormer top-k "
            "predicted RSS directions"
        ),
        "sweep_baseline": {
            "label": sweep_label,
            "mode": args.sweep_mode,
            "requested_sweep_size": args.sweep_size,
            "query_grid": "80 x 20 BeamFormer query beam grid",
        },
        "gain_loss_to_optimal_db": {
            label: baseline.summarize(values)
            for label, values in gain_loss_sets.items()
        },
        "similarity_to_optimal": {
            label: baseline.summarize(values)
            for label, values in similarity_sets.items()
        },
        "omp_gain_loss_minus_sweep_gain_loss_db": baseline.summarize(
            [row["omp_gain_loss_minus_sweep_gain_loss_db"] for row in metric_rows]
        ),
        "ebno_dbs": ebno_dbs.tolist(),
        "ber_mode": "monte_carlo",
        "ber_type": (
            "Monte Carlo uncoded QPSK using method RSS / optimal RSS as "
            "the Eb/N0 scale"
        ),
        "ber_symbols_per_sample": args.ber_symbols_per_sample,
        "ber": ber_curves,
    }

    write_csv(metric_rows, output_dir / "per_sample_metrics.csv")
    write_csv(topk_rows, output_dir / "topk_rss_directions.csv")
    write_json(summary, output_dir / "summary.json")
    save_vectors_npz(output_dir / "beam_vectors_topk_omp.npz", vector_payload)

    baseline.plot_ber(ebno_dbs, ber_curves, output_dir / "ber_comparison.png")

    all_gain_losses = [
        value for values in gain_loss_sets.values() for value in values
    ]
    gain_loss_xlim = (
        min(-0.05, float(np.min(all_gain_losses)) - 0.05),
        max(1.0, float(np.percentile(all_gain_losses, 99)) * 1.1),
    )
    baseline.plot_multi_cdf(
        gain_loss_sets,
        output_dir / "gain_loss_to_optimal_cdf.png",
        "Beamforming gain loss to optimal [dB]",
        "Gain Loss to Optimal Beam",
        xlim=gain_loss_xlim,
    )
    baseline.plot_multi_cdf(
        similarity_sets,
        output_dir / "similarity_to_optimal_cdf.png",
        "Normalized beam-vector similarity",
        "Similarity to Optimal Beam",
        xlim=(-0.02, 1.02),
    )

    print(f"\nWrote metrics CSV:       {output_dir / 'per_sample_metrics.csv'}")
    print(f"Wrote top-k RSS CSV:     {output_dir / 'topk_rss_directions.csv'}")
    print(f"Wrote vector NPZ:        {output_dir / 'beam_vectors_topk_omp.npz'}")
    print(f"Wrote summary JSON:      {output_dir / 'summary.json'}")
    print(f"Wrote BER plot:          {output_dir / 'ber_comparison.png'}")
    print(f"Wrote gain-loss CDF:     {output_dir / 'gain_loss_to_optimal_cdf.png'}")
    print(f"Sweep baseline:          {sweep_label}")
    print(
        "Median OMP loss [dB]:    "
        f"{summary['gain_loss_to_optimal_db'][f'Optimal vs {omp_label}']['median']:.4f}"
    )
    print(
        "Median sweep loss [dB]:  "
        f"{summary['gain_loss_to_optimal_db'][f'Optimal vs {sweep_label}']['median']:.4f}"
    )
    print(
        "Mean OMP similarity:     "
        f"{summary['similarity_to_optimal'][f'Optimal vs {omp_label}']['mean']:.4f}"
    )


if __name__ == "__main__":
    main()
