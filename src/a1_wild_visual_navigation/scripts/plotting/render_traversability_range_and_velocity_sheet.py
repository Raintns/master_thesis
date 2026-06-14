#!/usr/bin/env python3

import argparse
import csv
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


DEFAULT_VELOCITY_DIR = Path(
    "/home/rain/github_upload/Result/velocity_default(with_frame_output_detail)"
)
DEFAULT_FORCE_DIR = Path(
    "/home/rain/github_upload/Result/force_default_normalize"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/rain/github_upload/Result/rendered_figures"
)
DEFAULT_FRAMES = "420,700,1000,1520,2040,2420,2700,2820,3340"
DEFAULT_RANGE_START = 420
DEFAULT_RANGE_END = 3340
ROW_LABELS = ("Input", "Traversability", "Confidence")
PANEL_FILES = ("raw.png", "traversability_overlay.png", "confidence_overlay.png")


def load_font(size, bold=False):
    candidates = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            ]
        )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a traversability-range plot and a combined velocity frame sheet."
    )
    parser.add_argument("--velocity-dir", type=Path, default=DEFAULT_VELOCITY_DIR)
    parser.add_argument("--force-dir", type=Path, default=DEFAULT_FORCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--range-start", type=int, default=DEFAULT_RANGE_START)
    parser.add_argument("--range-end", type=int, default=DEFAULT_RANGE_END)
    parser.add_argument("--frames", default=DEFAULT_FRAMES)
    parser.add_argument("--panel-width", type=int, default=175)
    return parser.parse_args()


def parse_frames(text):
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def parse_optional_float(text):
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    return float(text)


def load_manifest_rows(manifest_path):
    with manifest_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def manifest_by_frame(rows):
    return {int(row["frame_index"]): row for row in rows}


def draw_multiline_text(draw, origin, lines, font, fill, line_spacing=2):
    x, y = origin
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y = bbox[3] + line_spacing


def load_panel(run_dir, frame_dir, panel_name, panel_width):
    image_path = run_dir / "wvn_frames" / frame_dir / panel_name
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    panel_height = round(panel_width * height / width)
    return image.resize((panel_width, panel_height), Image.Resampling.BILINEAR)


def rows_in_range(rows, range_start, range_end):
    selected = []
    for row in rows:
        frame_index = int(row["frame_index"])
        if range_start <= frame_index <= range_end:
            value = parse_optional_float(row.get("instant_traversability"))
            if value is not None:
                selected.append((frame_index, value))
    return selected


def scalar_rows_in_range(rows, range_start, range_end):
    selected = []
    for row in rows:
        frame_index = int(row["frame_index"])
        if range_start <= frame_index <= range_end:
            selected.append(row)
    return selected


def draw_axis_label_centered(draw, box_left, box_right, y, text, font, fill):
    bbox = draw.textbbox((0, 0), text, font=font)
    x = box_left + (box_right - box_left - (bbox[2] - bbox[0])) / 2
    draw.text((x, y), text, font=font, fill=fill)


def create_signal_plot(rows, representative_frames, range_start, range_end, title, line_color):
    points = rows_in_range(rows, range_start, range_end)
    if not points:
        return Image.new("RGB", (1400, 420), "white")

    x_values = [frame_index for frame_index, _ in points]
    y_values = [value for _, value in points]
    y_min_data = min(y_values)
    y_max_data = max(y_values)
    y_padding = max((y_max_data - y_min_data) * 0.12, 0.002)
    y_min = max(0.0, y_min_data - y_padding)
    y_max = min(1.0, y_max_data + y_padding)
    if y_max <= y_min:
        y_max = min(1.0, y_min + 0.01)

    fig, ax = plt.subplots(figsize=(14, 4.2), dpi=100)
    ax.plot(x_values, y_values, color=line_color, linewidth=1.6, label=title.replace(" vs Frame Index", ""))
    ax.set_xlim(range_start, range_end)
    ax.set_ylim(y_min, y_max)
    ax.set_title(title)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Traversability score τ")
    ax.grid(True, alpha=0.28)

    for index, frame_index in enumerate(representative_frames):
        if frame_index < range_start or frame_index > range_end:
            continue
        representative_value = None
        for point_x, point_y in points:
            if point_x == frame_index:
                representative_value = point_y
                break
        if representative_value is None:
            continue
        ax.axvline(frame_index, color="#bdbdbd", linewidth=0.7, alpha=0.7)
        ax.scatter(
            [frame_index],
            [representative_value],
            color="#d62728",
            edgecolors="white",
            linewidths=0.5,
            s=42,
            zorder=3,
        )
        y_offset = 12 if index % 2 == 0 else -18
        ax.annotate(
            str(frame_index),
            (frame_index, representative_value),
            textcoords="offset points",
            xytext=(0, y_offset),
            ha="center",
            color="#d62728",
            fontsize=10,
            fontweight="bold",
        )

    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def build_series(rows, value_key):
    xs = []
    ys = []
    for row in rows:
        value = parse_optional_float(row.get(value_key))
        if value is None:
            continue
        xs.append(int(row["frame_index"]))
        ys.append(value)
    return xs, ys


def create_traversability_error_curve(rows, range_start, range_end, title, error_key, filtered_key, error_label):
    selected_rows = scalar_rows_in_range(rows, range_start, range_end)
    traversability_x, traversability_y = build_series(selected_rows, "instant_traversability")
    error_x, error_y = build_series(selected_rows, error_key)
    filtered_x, filtered_y = build_series(selected_rows, filtered_key)

    if not traversability_x and not error_x and not filtered_x:
        return Image.new("RGB", (1800, 980), "white")

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 9),
        dpi=150,
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.08]},
    )
    fig.suptitle(title, fontsize=18, fontweight="normal")

    top_ax, bottom_ax = axes
    top_ax.plot(
        traversability_x,
        traversability_y,
        color="#1f77b4",
        linewidth=1.6,
        label="Instant Traversability",
    )
    top_ax.set_title("Instant Traversability", fontsize=14)
    top_ax.set_ylabel("Traversability score τ")
    top_ax.grid(True, which="major", alpha=0.45)
    top_ax.grid(True, which="minor", alpha=0.2, linestyle=":")
    top_ax.minorticks_on()
    top_ax.legend(loc="lower left", frameon=True)

    bottom_ax.plot(error_x, error_y, color="#ff7f0e", linewidth=1.6, label=f"{error_label} MSE")
    bottom_ax.plot(
        filtered_x,
        filtered_y,
        color="#2ca02c",
        linewidth=1.6,
        linestyle="--",
        label=f"{error_label} filtered",
    )
    bottom_ax.set_title(f"{error_label}", fontsize=14)
    bottom_ax.set_xlabel("Frame Index")
    bottom_ax.set_ylabel(f"{error_label}")
    bottom_ax.grid(True, which="major", alpha=0.45)
    bottom_ax.grid(True, which="minor", alpha=0.2, linestyle=":")
    bottom_ax.minorticks_on()
    bottom_ax.legend(loc="upper left", frameon=True)

    fig.tight_layout(rect=(0, 0, 1, 0.965))

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def create_frame_sheet(run_dir, rows_by_frame, representative_frames, panel_width, title, error_field, error_label):
    sample_row = rows_by_frame[representative_frames[0]]
    sample_image = load_panel(run_dir, sample_row["frame_dir"], PANEL_FILES[0], panel_width)
    panel_height = sample_image.height

    title_font = load_font(28, bold=True)
    text_font = load_font(17)
    section_font = load_font(18, bold=True)

    margin = 34
    col_gap = 14
    row_gap = 18
    header_height = 78
    row_label_width = 150

    width = margin * 2 + row_label_width + len(representative_frames) * panel_width + (len(representative_frames) - 1) * col_gap
    height = margin * 2 + 40 + header_height + 3 * panel_height + 2 * row_gap

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw_axis_label_centered(draw, 0, width, 10, title, title_font, "black")

    content_x = margin + row_label_width
    annotation_y = margin + 36

    for col_index, frame_index in enumerate(representative_frames):
        frame_x = content_x + col_index * (panel_width + col_gap)
        row = rows_by_frame[frame_index]
        traversability = parse_optional_float(row.get("instant_traversability"))
        error_value = parse_optional_float(row.get(error_field))
        lines = [f"Frame {frame_index}"]
        if traversability is not None:
            lines.append(f"T = {traversability:.4f}")
        if error_value is not None:
            lines.append(f"{error_label} = {error_value:.6f}")
        draw_multiline_text(draw, (frame_x, annotation_y), lines, text_font, "black")

    image_top = annotation_y + header_height
    for row_index, (row_label, panel_name) in enumerate(zip(ROW_LABELS, PANEL_FILES)):
        row_y = image_top + row_index * (panel_height + row_gap)
        label_bbox = draw.textbbox((0, 0), row_label, font=section_font)
        label_y = row_y + (panel_height - (label_bbox[3] - label_bbox[1])) // 2
        draw.text((margin, label_y), row_label, font=section_font, fill="black")
        for col_index, frame_index in enumerate(representative_frames):
            frame_x = content_x + col_index * (panel_width + col_gap)
            panel_image = load_panel(run_dir, rows_by_frame[frame_index]["frame_dir"], panel_name, panel_width)
            canvas.paste(panel_image, (frame_x, row_y))

    return canvas


def main():
    args = parse_args()
    representative_frames = parse_frames(args.frames)

    velocity_manifest_rows = load_manifest_rows(args.velocity_dir / "wvn_frames" / "manifest.csv")
    force_manifest_rows = load_manifest_rows(args.force_dir / "wvn_frames" / "manifest.csv")
    velocity_by_frame = manifest_by_frame(velocity_manifest_rows)
    force_by_frame = manifest_by_frame(force_manifest_rows)
    missing_velocity = [frame for frame in representative_frames if frame not in velocity_by_frame]
    if missing_velocity:
        raise KeyError(f"Representative velocity frames missing: {missing_velocity}")
    missing_force = [frame for frame in representative_frames if frame not in force_by_frame]
    if missing_force:
        raise KeyError(f"Representative force frames missing: {missing_force}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    velocity_plot = create_signal_plot(
        rows=velocity_manifest_rows,
        representative_frames=representative_frames,
        range_start=args.range_start,
        range_end=args.range_end,
        title="Velocity Traversability Over Time",
        line_color="#1f77b4",
    )
    force_plot = create_signal_plot(
        rows=force_manifest_rows,
        representative_frames=representative_frames,
        range_start=args.range_start,
        range_end=args.range_end,
        title="Force Traversability Over Time",
        line_color="#1f77b4",
    )
    velocity_curve_plot = create_traversability_error_curve(
        rows=velocity_manifest_rows,
        range_start=args.range_start,
        range_end=args.range_end,
        title="Traversability and Velocity Error Over Time",
        error_key="velocity_error_mse",
        filtered_key="velocity_error_filtered",
        error_label="Velocity error",
    )
    force_curve_plot = create_traversability_error_curve(
        rows=force_manifest_rows,
        range_start=args.range_start,
        range_end=args.range_end,
        title="Traversability and Force Error Over Time",
        error_key="force_error_mse",
        filtered_key="force_error_filtered",
        error_label="Force error",
    )
    velocity_sheet = create_frame_sheet(
        run_dir=args.velocity_dir,
        rows_by_frame=velocity_by_frame,
        representative_frames=representative_frames,
        panel_width=args.panel_width,
        title="Velocity-Based Representative Frames",
        error_field="velocity_error_filtered",
        error_label="vel err",
    )
    force_sheet = create_frame_sheet(
        run_dir=args.force_dir,
        rows_by_frame=force_by_frame,
        representative_frames=representative_frames,
        panel_width=args.panel_width,
        title="Force-Based Representative Frames",
        error_field="force_error_filtered",
        error_label="force err",
    )

    velocity_plot_path = args.output_dir / "velocity_traversability_0420_3340_styled.png"
    force_plot_path = args.output_dir / "force_traversability_0420_3340_styled.png"
    velocity_curve_path = args.output_dir / "traversability_velocity_curves_0420_3340.png"
    force_curve_path = args.output_dir / "traversability_force_curves_0420_3340.png"
    velocity_sheet_path = args.output_dir / "velocity_frames_0420_3340.png"
    force_sheet_path = args.output_dir / "force_frames_0420_3340.png"

    velocity_plot.save(velocity_plot_path)
    force_plot.save(force_plot_path)
    velocity_curve_plot.save(velocity_curve_path)
    force_curve_plot.save(force_curve_path)
    velocity_sheet.save(velocity_sheet_path)
    force_sheet.save(force_sheet_path)

    print(f"[ok] wrote {velocity_plot_path}")
    print(f"[ok] wrote {force_plot_path}")
    print(f"[ok] wrote {velocity_curve_path}")
    print(f"[ok] wrote {force_curve_path}")
    print(f"[ok] wrote {velocity_sheet_path}")
    print(f"[ok] wrote {force_sheet_path}")


if __name__ == "__main__":
    main()
