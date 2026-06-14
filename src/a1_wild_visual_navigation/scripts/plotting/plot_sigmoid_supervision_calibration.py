#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_VELOCITY_CSV = Path(
    "/home/rain/github_upload/Result/velocity_default(with_frame_output_detail)/wvn_velocity_debug.csv"
)
DEFAULT_FORCE_CSV = Path("/home/rain/github_upload/Result/force_default_normalize/wvn_force_debug.csv")
DEFAULT_VELOCITY_MANIFEST = Path(
    "/home/rain/github_upload/Result/velocity_default(with_frame_output_detail)/wvn_frames/manifest.csv"
)
DEFAULT_FORCE_MANIFEST = Path(
    "/home/rain/github_upload/Result/force_default_normalize/wvn_frames/manifest.csv"
)
DEFAULT_OUTPUT_DIR = Path("/home/rain/github_upload/Result/rendered_figures")
DEFAULT_OUTPUT_NAME = "sigmoid_supervision_calibration.png"
DEFAULT_SUMMARY_NAME = "sigmoid_supervision_calibration_summary.csv"
DEFAULT_FORMULA_NAME = "sigmoid_candidate_formula_table.tex"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot supervision-sigmoid calibration for velocity and force error signals."
    )
    parser.add_argument("--velocity-csv", type=Path, default=DEFAULT_VELOCITY_CSV)
    parser.add_argument("--force-csv", type=Path, default=DEFAULT_FORCE_CSV)
    parser.add_argument("--velocity-manifest", type=Path, default=DEFAULT_VELOCITY_MANIFEST)
    parser.add_argument("--force-manifest", type=Path, default=DEFAULT_FORCE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--current-slope", type=float, default=20.0)
    parser.add_argument("--current-cutoff", type=float, default=0.25)
    parser.add_argument("--target-high", type=float, default=0.95)
    parser.add_argument("--target-low", type=float, default=0.05)
    parser.add_argument("--lower-percentile", type=float, default=0.05)
    parser.add_argument("--upper-percentile", type=float, default=0.95)
    parser.add_argument("--range-start", type=int, default=420)
    parser.add_argument("--range-end", type=int, default=3340)
    return parser.parse_args()


def load_manifest(path):
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "frame_index": int(row["frame_index"]),
                    "timestamp": int(row["stamp_secs"]) + int(row["stamp_nsecs"]) * 1e-9,
                }
            )
    rows.sort(key=lambda item: item["frame_index"])
    return rows


def interpolate_timestamp(manifest_rows, frame_index):
    if frame_index <= manifest_rows[0]["frame_index"]:
        return manifest_rows[0]["timestamp"]
    if frame_index >= manifest_rows[-1]["frame_index"]:
        return manifest_rows[-1]["timestamp"]

    for left, right in zip(manifest_rows[:-1], manifest_rows[1:]):
        f0 = left["frame_index"]
        f1 = right["frame_index"]
        if f0 <= frame_index <= f1:
            if f1 == f0:
                return left["timestamp"]
            alpha = (frame_index - f0) / float(f1 - f0)
            return left["timestamp"] + alpha * (right["timestamp"] - left["timestamp"])

    raise ValueError(f"Could not interpolate timestamp for frame {frame_index}")


def read_csv_column(path, key, time_start=None, time_end=None):
    values = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if time_start is not None or time_end is not None:
                ts_value = row.get("timestamp", "")
                if ts_value in ("", None):
                    continue
                timestamp = float(ts_value)
                if time_start is not None and timestamp < time_start:
                    continue
                if time_end is not None and timestamp > time_end:
                    continue
            value = row.get(key, "")
            if value in ("", None):
                continue
            values.append(float(value))
    return np.asarray(values, dtype=float)


def percentile(values, p):
    return float(np.quantile(values, p))


def percentile_label(p):
    return f"p{int(round(p * 100)):02d}"


def percentile_quantile_tex(p):
    return f"{p:.2f}"


def ordinal_label(n):
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def sigmoid(error_values, slope, cutoff):
    return 1.0 / (1.0 + np.exp(slope * (error_values - cutoff)))


def fit_sigmoid_from_band(error_low, error_high, target_high, target_low):
    logit = lambda t: math.log(t / (1.0 - t))
    cutoff = 0.5 * (error_low + error_high)
    slope = abs((logit(target_high) - logit(target_low)) / (error_high - error_low))
    return slope, cutoff


def describe_distribution(values):
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "p05": percentile(values, 0.05),
        "p25": percentile(values, 0.25),
        "median": percentile(values, 0.5),
        "p75": percentile(values, 0.75),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def build_case(
    label,
    csv_path,
    error_key,
    tau_key,
    current_slope,
    current_cutoff,
    low_p,
    high_p,
    target_high,
    target_low,
    time_start=None,
    time_end=None,
):
    errors = read_csv_column(csv_path, error_key, time_start=time_start, time_end=time_end)
    taus = read_csv_column(csv_path, tau_key, time_start=time_start, time_end=time_end)
    error_low = percentile(errors, low_p)
    error_high = percentile(errors, high_p)
    candidate_slope, candidate_cutoff = fit_sigmoid_from_band(error_low, error_high, target_high, target_low)
    current_tau = sigmoid(errors, current_slope, current_cutoff)
    candidate_tau = sigmoid(errors, candidate_slope, candidate_cutoff)

    return {
        "label": label,
        "error_key": error_key,
        "tau_key": tau_key,
        "errors": errors,
        "taus": taus,
        "current_slope": current_slope,
        "current_cutoff": current_cutoff,
        "candidate_slope": candidate_slope,
        "candidate_cutoff": candidate_cutoff,
        "error_stats": describe_distribution(errors),
        "tau_stats": describe_distribution(taus),
        "error_low": error_low,
        "error_high": error_high,
        "current_tau": current_tau,
        "candidate_tau": candidate_tau,
    }


def write_summary(cases, output_path):
    fieldnames = [
        "case",
        "error_key",
        "tau_key",
        "count",
        "error_min",
        "error_p05",
        "error_p25",
        "error_median",
        "error_p75",
        "error_p95",
        "error_max",
        "error_mean",
        "error_std",
        "error_p99",
        "tau_min",
        "tau_p05",
        "tau_p25",
        "tau_median",
        "tau_p75",
        "tau_p95",
        "tau_max",
        "tau_mean",
        "tau_std",
        "current_slope",
        "current_cutoff",
        "candidate_slope",
        "candidate_cutoff",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "case": case["label"],
                    "error_key": case["error_key"],
                    "tau_key": case["tau_key"],
                    "count": case["error_stats"]["count"],
                    "error_min": case["error_stats"]["min"],
                    "error_p05": case["error_stats"]["p05"],
                    "error_p25": case["error_stats"]["p25"],
                    "error_median": case["error_stats"]["median"],
                    "error_p75": case["error_stats"]["p75"],
                    "error_p95": case["error_stats"]["p95"],
                    "error_max": case["error_stats"]["max"],
                    "error_mean": case["error_stats"]["mean"],
                    "error_std": case["error_stats"]["std"],
                    "error_p99": case["error_stats"]["p99"],
                    "tau_min": case["tau_stats"]["min"],
                    "tau_p05": case["tau_stats"]["p05"],
                    "tau_p25": case["tau_stats"]["p25"],
                    "tau_median": case["tau_stats"]["median"],
                    "tau_p75": case["tau_stats"]["p75"],
                    "tau_p95": case["tau_stats"]["p95"],
                    "tau_max": case["tau_stats"]["max"],
                    "tau_mean": case["tau_stats"]["mean"],
                    "tau_std": case["tau_stats"]["std"],
                    "current_slope": case["current_slope"],
                    "current_cutoff": case["current_cutoff"],
                    "candidate_slope": case["candidate_slope"],
                    "candidate_cutoff": case["candidate_cutoff"],
                }
            )


def plot_case_row(axes_row, case, low_p, high_p):
    hist_ax, mapping_ax, tau_ax = axes_row
    errors = case["errors"]
    taus = case["taus"]
    current_tau = case["current_tau"]
    candidate_tau = case["candidate_tau"]
    low_label = percentile_label(low_p)
    high_label = percentile_label(high_p)

    x_max = max(
        case["error_stats"]["p99"] * 1.05,
        case["candidate_cutoff"] * 1.35,
        case["error_high"] * 1.05,
    )
    x_values = np.linspace(0.0, x_max, 600)
    current_curve = sigmoid(x_values, case["current_slope"], case["current_cutoff"])
    candidate_curve = sigmoid(x_values, case["candidate_slope"], case["candidate_cutoff"])

    hist_ax.hist(errors, bins=30, color="#4c78a8", alpha=0.75, edgecolor="white")
    hist_ax.axvline(case["error_low"], color="#54a24b", linestyle="--", linewidth=1.2, label=low_label)
    hist_ax.axvline(case["error_high"], color="#e45756", linestyle="--", linewidth=1.2, label=high_label)
    hist_ax.axvline(case["candidate_cutoff"], color="#b279a2", linewidth=1.3, label="Candidate cutoff")
    hist_ax.set_title(f"{case['label']}: Filtered Error Distribution", fontsize=12)
    hist_ax.set_xlabel("Filtered error")
    hist_ax.set_ylabel("Sample count")
    hist_ax.set_xlim(0.0, x_max)
    hist_ax.grid(True, which="major", alpha=0.35)
    hist_ax.grid(True, which="minor", alpha=0.2, linestyle=":")
    hist_ax.minorticks_on()
    hist_ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
    hist_legend = hist_ax.legend(
        loc="upper right",
        fontsize=8,
        frameon=True,
        title=(
            f"Current cutoff: {case['current_cutoff']:.4f}"
            + (" (off-range)" if case["current_cutoff"] > x_max else "")
            + f"\nCandidate cutoff: {case['candidate_cutoff']:.4f}"
        ),
        title_fontsize=8,
    )
    hist_legend._legend_box.align = "left"

    mapping_ax.plot(
        x_values,
        current_curve,
        color="#f58518",
        linewidth=2.0,
        label=f"Current: s={case['current_slope']:.0f}, c={case['current_cutoff']:.4f}",
    )
    mapping_ax.plot(
        x_values,
        candidate_curve,
        color="#b279a2",
        linewidth=2.0,
        label=f"Candidate: s={case['candidate_slope']:.1f}, c={case['candidate_cutoff']:.4f}",
    )
    mapping_ax.axvline(case["error_low"], color="#54a24b", linestyle="--", linewidth=1.2)
    mapping_ax.axvline(case["error_high"], color="#e45756", linestyle="--", linewidth=1.2)
    mapping_ax.set_title(f"{case['label']}: Error-to-Traversability Mapping", fontsize=12)
    mapping_ax.set_xlabel("Filtered error")
    mapping_ax.set_ylabel("Traversability score τ")
    mapping_ax.set_ylim(-0.02, 1.02)
    mapping_ax.set_xlim(0.0, x_max)
    mapping_ax.grid(True, which="major", alpha=0.35)
    mapping_ax.grid(True, which="minor", alpha=0.2, linestyle=":")
    mapping_ax.minorticks_on()
    mapping_ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
    mapping_ax.legend(loc="lower left", fontsize=8, frameon=True)

    tau_bins = np.linspace(0.0, 1.0, 61)
    tau_ax.hist(
        current_tau,
        bins=tau_bins,
        alpha=0.28,
        color="#f58518",
        edgecolor="none",
        label="Current mapping τ",
    )
    tau_ax.hist(
        candidate_tau,
        bins=tau_bins,
        alpha=0.35,
        color="#b279a2",
        edgecolor="none",
        label="Candidate mapping τ",
    )
    tau_ax.hist(
        taus,
        bins=tau_bins,
        histtype="step",
        linewidth=2.2,
        color="#4c78a8",
        label="Recorded τ",
        zorder=4,
    )
    tau_ax.set_title(f"{case['label']}: Resulting Traversability Distribution", fontsize=12)
    tau_ax.set_xlabel("Traversability score τ")
    tau_ax.set_ylabel("Sample count")
    tau_ax.grid(True, which="major", alpha=0.35)
    tau_ax.grid(True, which="minor", alpha=0.2, linestyle=":")
    tau_ax.minorticks_on()
    tau_ax.legend(loc="upper left", fontsize=8, frameon=True)

    # Keep the original inset style, but shift it left so it does not sit on top
    # of the sharp high-tau peak at the right edge of the main histogram.
    inset = tau_ax.inset_axes([0.36, 0.38, 0.42, 0.38])
    zoom_tau_min = min(float(np.min(taus)), float(np.min(current_tau)))
    zoom_tau_max = max(float(np.max(taus)), float(np.max(current_tau)))
    pad = max((zoom_tau_max - zoom_tau_min) * 0.2, 0.00008)
    inset.hist(current_tau, bins=25, histtype="step", linewidth=1.5, color="#f58518")
    inset.hist(taus, bins=25, histtype="step", linewidth=1.9, color="#4c78a8")
    inset.set_xlim(max(0.0, zoom_tau_min - pad), min(1.0, zoom_tau_max + pad))
    inset.set_title("Recorded vs current", fontsize=8)
    inset.grid(True, alpha=0.25)
    inset.tick_params(axis="both", labelsize=7)


def plot_cases(cases, output_path, low_p, high_p):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5), dpi=160)
    fig.suptitle("Supervision Sigmoid Calibration: Error Range, Mapping, and Label Spread", fontsize=18)

    for axes_row, case in zip(axes, cases):
        plot_case_row(axes_row, case, low_p, high_p)

    fig.tight_layout(rect=(0, 0, 1, 0.965))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="white")
    plt.close(fig)


def write_formula_table(cases, output_path, low_p, high_p, target_high, target_low):
    by_label = {case["label"]: case for case in cases}
    velocity = by_label["Velocity"]
    force = by_label["Force"]
    low_q = percentile_quantile_tex(low_p)
    high_q = percentile_quantile_tex(high_p)
    low_pct = int(round(low_p * 100))
    high_pct = int(round(high_p * 100))
    low_ord = ordinal_label(low_pct)
    high_ord = ordinal_label(high_pct)
    logit_constant = math.log(target_high / (1.0 - target_high)) - math.log(target_low / (1.0 - target_low))
    text = f"""\\begin{{table}}[t]
\\centering
\\caption{{Computation of the candidate sigmoid cutoff and slope from the filtered-error distribution for the analyzed frame window using the {low_ord}--{high_ord} percentile error band.}}
\\label{{tab:sigmoid_candidate_formula}}
\\begin{{tabular}}{{lll}}
\\toprule
Step & Formula & Meaning \\\\
\\midrule
Lower error bound & $e_{{\\mathrm{{low}}}} = Q_{{{low_q}}}(e)$ & {low_ord} percentile of the filtered error \\\\
Upper error bound & $e_{{\\mathrm{{high}}}} = Q_{{{high_q}}}(e)$ & {high_ord} percentile of the filtered error \\\\
Candidate cutoff & $c_{{\\mathrm{{cand}}}} = \\dfrac{{e_{{\\mathrm{{low}}}} + e_{{\\mathrm{{high}}}}}}{{2}}$ & midpoint of the fitted error band \\\\
Candidate slope & $s_{{\\mathrm{{cand}}}} = \\dfrac{{\\logit(\\tau_{{\\mathrm{{high}}}}) - \\logit(\\tau_{{\\mathrm{{low}}}})}}{{e_{{\\mathrm{{high}}}} - e_{{\\mathrm{{low}}}}}}$ & slope fitted to the chosen traversability band \\\\
Target labels & $\\tau_{{\\mathrm{{high}}}} = {target_high:.2f},\\; \\tau_{{\\mathrm{{low}}}} = {target_low:.2f}$ & same targets used in the calibration script \\\\
Equivalent constant & $s_{{\\mathrm{{cand}}}} \\approx \\dfrac{{{logit_constant:.4f}}}{{e_{{\\mathrm{{high}}}} - e_{{\\mathrm{{low}}}}}}$ & since $\\logit({target_high:.2f})-\\logit({target_low:.2f})={logit_constant:.4f}$ \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\begin{{table}}[t]
\\centering
\\caption{{Numeric candidate-cutoff and candidate-slope calculation for the analyzed velocity- and force-based supervision signals in frames 420--3340 using the {low_ord}--{high_ord} percentile error band.}}
\\label{{tab:sigmoid_candidate_values}}
\\begin{{tabular}}{{lcc}}
\\toprule
Quantity & Velocity & Force \\\\
\\midrule
$e_{{\\mathrm{{low}}}} = Q_{{{low_q}}}(e)$ & ${velocity['error_low']:.7f}$ & ${force['error_low']:.7f}$ \\\\
$e_{{\\mathrm{{high}}}} = Q_{{{high_q}}}(e)$ & ${velocity['error_high']:.7f}$ & ${force['error_high']:.7f}$ \\\\
$c_{{\\mathrm{{cand}}}} = \\dfrac{{e_{{\\mathrm{{low}}}} + e_{{\\mathrm{{high}}}}}}{{2}}$ & $\\dfrac{{{velocity['error_low']:.7f} + {velocity['error_high']:.7f}}}{{2}} = {velocity['candidate_cutoff']:.7f}$ & $\\dfrac{{{force['error_low']:.7f} + {force['error_high']:.7f}}}{{2}} = {force['candidate_cutoff']:.7f}$ \\\\
$s_{{\\mathrm{{cand}}}} = \\dfrac{{{logit_constant:.4f}}}{{e_{{\\mathrm{{high}}}} - e_{{\\mathrm{{low}}}}}}$ & $\\dfrac{{{logit_constant:.4f}}}{{{velocity['error_high']:.7f} - {velocity['error_low']:.7f}}} = {velocity['candidate_slope']:.2f}$ & $\\dfrac{{{logit_constant:.4f}}}{{{force['error_high']:.7f} - {force['error_low']:.7f}}} = {force['candidate_slope']:.2f}$ \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    output_path.write_text(text)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    velocity_manifest = load_manifest(args.velocity_manifest)
    force_manifest = load_manifest(args.force_manifest)
    velocity_t0 = interpolate_timestamp(velocity_manifest, args.range_start)
    velocity_t1 = interpolate_timestamp(velocity_manifest, args.range_end)
    force_t0 = interpolate_timestamp(force_manifest, args.range_start)
    force_t1 = interpolate_timestamp(force_manifest, args.range_end)

    cases = [
        build_case(
            label="Velocity",
            csv_path=args.velocity_csv,
            error_key="velocity_tracking_error_filtered",
            tau_key="supervision_traversability",
            current_slope=args.current_slope,
            current_cutoff=args.current_cutoff,
            low_p=args.lower_percentile,
            high_p=args.upper_percentile,
            target_high=args.target_high,
            target_low=args.target_low,
            time_start=velocity_t0,
            time_end=velocity_t1,
        ),
        build_case(
            label="Force",
            csv_path=args.force_csv,
            error_key="force_error_filtered",
            tau_key="supervision_traversability",
            current_slope=args.current_slope,
            current_cutoff=args.current_cutoff,
            low_p=args.lower_percentile,
            high_p=args.upper_percentile,
            target_high=args.target_high,
            target_low=args.target_low,
            time_start=force_t0,
            time_end=force_t1,
        ),
    ]

    output_path = args.output_dir / DEFAULT_OUTPUT_NAME
    summary_path = args.output_dir / DEFAULT_SUMMARY_NAME
    formula_path = args.output_dir / DEFAULT_FORMULA_NAME
    plot_cases(cases, output_path, args.lower_percentile, args.upper_percentile)
    write_summary(cases, summary_path)
    write_formula_table(
        cases,
        formula_path,
        args.lower_percentile,
        args.upper_percentile,
        args.target_high,
        args.target_low,
    )

    print(f"[ok] wrote {output_path}")
    print(f"[ok] wrote {summary_path}")
    print(f"[ok] wrote {formula_path}")
    for case in cases:
        print(
            f"[{case['label'].lower()}] current s={case['current_slope']:.1f}, c={case['current_cutoff']:.4f}; "
            f"candidate s={case['candidate_slope']:.2f}, c={case['candidate_cutoff']:.6f}"
        )


if __name__ == "__main__":
    main()
