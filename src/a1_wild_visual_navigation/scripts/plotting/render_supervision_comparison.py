#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

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
DEFAULT_FRAME_GROUPS = "420,700,1000,1520;2040,2420,2700;2820,3340"
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
        description="Render comparison figures for velocity- and force-based supervision runs."
    )
    parser.add_argument(
        "--velocity-dir",
        type=Path,
        default=DEFAULT_VELOCITY_DIR,
        help="Velocity result directory containing wvn_frames/manifest.csv.",
    )
    parser.add_argument(
        "--force-dir",
        type=Path,
        default=DEFAULT_FORCE_DIR,
        help="Force result directory containing wvn_frames/manifest.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where rendered figures will be written.",
    )
    parser.add_argument(
        "--frame-groups",
        default=DEFAULT_FRAME_GROUPS,
        help="Semicolon-separated groups of comma-separated frame indices.",
    )
    parser.add_argument(
        "--panel-width",
        type=int,
        default=240,
        help="Width for each rendered panel in pixels.",
    )
    return parser.parse_args()


def parse_frame_groups(frame_groups_text):
    groups = []
    for group_text in frame_groups_text.split(";"):
        group_text = group_text.strip()
        if not group_text:
            continue
        groups.append([int(value.strip()) for value in group_text.split(",") if value.strip()])
    if not groups:
        raise ValueError("No frame groups were provided.")
    return groups


def load_manifest_by_frame(manifest_path):
    with manifest_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapping = {}
    for row in rows:
        mapping[int(row["frame_index"])] = row
    return mapping


def parse_optional_float(text):
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    return float(text)


def load_panel(run_dir, frame_dir, panel_name, panel_width):
    image_path = run_dir / "wvn_frames" / frame_dir / panel_name
    image = Image.open(image_path).convert("RGB")
    original_width, original_height = image.size
    panel_height = round(panel_width * original_height / original_width)
    return image.resize((panel_width, panel_height), Image.Resampling.BILINEAR)


def build_traversability_plot(manifest, frame_group, width, height, line_color, text_font):
    plot = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(plot)

    left = 60
    right = width - 18
    top = 18
    bottom = height - 36

    rows = [manifest[idx] for idx in sorted(manifest.keys())]
    points = []
    for row in rows:
        value = parse_optional_float(row.get("instant_traversability"))
        if value is None:
            continue
        points.append((int(row["frame_index"]), value))

    if not points:
        return plot

    x_min = points[0][0]
    x_max = points[-1][0]
    value_min = min(value for _, value in points)
    value_max = max(value for _, value in points)
    value_range = value_max - value_min
    padding = max(value_range * 0.15, 0.002)
    y_min = max(0.0, value_min - padding)
    y_max = min(1.0, value_max + padding)
    if y_max <= y_min:
        y_max = min(1.0, y_min + 0.01)

    def to_xy(frame_index, value):
        if x_max == x_min:
            x = (left + right) / 2.0
        else:
            x = left + (frame_index - x_min) * (right - left) / (x_max - x_min)
        y = bottom - (value - y_min) * (bottom - top) / (y_max - y_min)
        return x, y

    draw.rounded_rectangle((left, top, right, bottom), radius=8, fill="#fafafa", outline="#bbbbbb", width=1)

    for y_value in (y_min, (y_min + y_max) / 2.0, y_max):
        y = to_xy(x_min, y_value)[1]
        draw.line((left, y, right, y), fill="#d9d9d9", width=1)
        label = f"{y_value:.4f}"
        bbox = draw.textbbox((0, 0), label, font=text_font)
        draw.text((left - 10 - (bbox[2] - bbox[0]), y - (bbox[3] - bbox[1]) / 2), label, font=text_font, fill="black")

    draw.line((left, top, left, bottom), fill="black", width=2)
    draw.line((left, bottom, right, bottom), fill="black", width=2)

    polyline = [to_xy(frame_index, value) for frame_index, value in points]
    for start, end in zip(polyline[:-1], polyline[1:]):
        draw.line((start[0], start[1], end[0], end[1]), fill=line_color, width=3)

    for frame_index in frame_group:
        row = manifest.get(frame_index)
        if row is None:
            continue
        value = parse_optional_float(row.get("instant_traversability"))
        if value is None:
            continue
        x, y = to_xy(frame_index, value)
        draw.line((x, top, x, bottom), fill="#bbbbbb", width=1)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#d62728", outline="white", width=1)
        label = str(frame_index)
        bbox = draw.textbbox((0, 0), label, font=text_font)
        label_x = min(max(x - (bbox[2] - bbox[0]) / 2, left), right - (bbox[2] - bbox[0]))
        label_y = max(top, y - 20 - (bbox[3] - bbox[1]))
        draw.text((label_x, label_y), label, font=text_font, fill="#d62728")

    title = "Traversability over saved frames (zoomed)"
    bbox = draw.textbbox((0, 0), title, font=text_font)
    draw.text(((width - (bbox[2] - bbox[0])) / 2, 0), title, font=text_font, fill="black")

    x_label = "Frame index"
    bbox = draw.textbbox((0, 0), x_label, font=text_font)
    draw.text(((left + right - (bbox[2] - bbox[0])) / 2, height - 24), x_label, font=text_font, fill="black")

    y_label = "T"
    bbox = draw.textbbox((0, 0), y_label, font=text_font)
    draw.text((10, (top + bottom - (bbox[3] - bbox[1])) / 2), y_label, font=text_font, fill="black")

    return plot


def build_mode_label(mode_name, frame_index, manifest_row):
    traversability = parse_optional_float(manifest_row.get("instant_traversability"))
    if mode_name == "Velocity":
        error = parse_optional_float(manifest_row.get("velocity_error_filtered"))
        error_label = "vel err"
    else:
        error = parse_optional_float(manifest_row.get("force_error_filtered"))
        error_label = "force err"

    lines = [f"Frame {frame_index}"]
    if traversability is not None:
        lines.append(f"T = {traversability:.4f}")
    if error is not None:
        lines.append(f"{error_label} = {error:.6f}")
    return lines


def draw_multiline_text(draw, origin, lines, font, fill, line_spacing):
    x, y = origin
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y = bbox[3] + line_spacing


def render_group(
    velocity_dir,
    force_dir,
    velocity_manifest,
    force_manifest,
    frame_group,
    output_path,
    panel_width,
):
    sample_row = velocity_manifest[frame_group[0]]
    sample_image = load_panel(velocity_dir, sample_row["frame_dir"], PANEL_FILES[0], panel_width)
    panel_height = sample_image.height

    title_font = load_font(26, bold=True)
    section_font = load_font(22, bold=True)
    text_font = load_font(17, bold=False)

    margin = 36
    section_gap = 34
    row_gap = 18
    plot_gap = 18
    col_gap = 16
    row_label_width = 150
    column_text_height = 78
    section_title_height = 34
    plot_height = 180
    group_width = len(frame_group) * panel_width + (len(frame_group) - 1) * col_gap
    row_height = panel_height
    section_height = (
        section_title_height
        + plot_height
        + plot_gap
        + column_text_height
        + 3 * row_height
        + 2 * row_gap
    )
    canvas_width = margin * 2 + row_label_width + group_width
    canvas_height = margin * 2 + section_height * 2 + section_gap + 44

    canvas = Image.new("RGB", (canvas_width, canvas_height), color="white")
    draw = ImageDraw.Draw(canvas)

    title = "Velocity-Based vs Force-Based Supervision"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    draw.text(
        ((canvas_width - (title_bbox[2] - title_bbox[0])) // 2, margin // 2),
        title,
        font=title_font,
        fill="black",
    )

    sections = [
        ("Velocity", velocity_dir, velocity_manifest, "#1f77b4"),
        ("Force", force_dir, force_manifest, "#ff7f0e"),
    ]

    content_x = margin + row_label_width
    current_y = margin + 26

    for section_name, run_dir, manifest, line_color in sections:
        draw.text((margin, current_y), section_name, font=section_font, fill="black")

        plot_top = current_y + section_title_height
        plot_image = build_traversability_plot(
            manifest=manifest,
            frame_group=frame_group,
            width=group_width,
            height=plot_height,
            line_color=line_color,
            text_font=text_font,
        )
        plot_label_bbox = draw.textbbox((0, 0), "Plot", font=text_font)
        plot_label_y = plot_top + (plot_height - (plot_label_bbox[3] - plot_label_bbox[1])) // 2
        draw.text((margin, plot_label_y), "Plot", font=text_font, fill="black")
        canvas.paste(plot_image, (content_x, plot_top))

        annotation_y = plot_top + plot_height + plot_gap
        for col_index, frame_index in enumerate(frame_group):
            frame_x = content_x + col_index * (panel_width + col_gap)
            manifest_row = manifest[frame_index]
            annotation_lines = build_mode_label(section_name, frame_index, manifest_row)
            draw_multiline_text(
                draw,
                (frame_x, annotation_y),
                annotation_lines,
                text_font,
                "black",
                line_spacing=2,
            )

        image_top = annotation_y + column_text_height
        for row_index, (row_label, panel_name) in enumerate(zip(ROW_LABELS, PANEL_FILES)):
            row_y = image_top + row_index * (row_height + row_gap)
            label_bbox = draw.textbbox((0, 0), row_label, font=text_font)
            label_y = row_y + (row_height - (label_bbox[3] - label_bbox[1])) // 2
            draw.text((margin, label_y), row_label, font=text_font, fill="black")

            for col_index, frame_index in enumerate(frame_group):
                frame_x = content_x + col_index * (panel_width + col_gap)
                frame_dir = manifest[frame_index]["frame_dir"]
                panel_image = load_panel(run_dir, frame_dir, panel_name, panel_width)
                canvas.paste(panel_image, (frame_x, row_y))

        current_y += section_height + section_gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main():
    args = parse_args()
    frame_groups = parse_frame_groups(args.frame_groups)

    velocity_manifest = load_manifest_by_frame(args.velocity_dir / "wvn_frames" / "manifest.csv")
    force_manifest = load_manifest_by_frame(args.force_dir / "wvn_frames" / "manifest.csv")

    for frame_group in frame_groups:
        missing_velocity = [idx for idx in frame_group if idx not in velocity_manifest]
        missing_force = [idx for idx in frame_group if idx not in force_manifest]
        if missing_velocity or missing_force:
            raise KeyError(
                f"Missing frames for group {frame_group}: "
                f"velocity={missing_velocity}, force={missing_force}"
            )

        group_slug = "_".join(f"{idx:04d}" for idx in frame_group)
        output_path = args.output_dir / f"comparison_{group_slug}.png"
        render_group(
            velocity_dir=args.velocity_dir,
            force_dir=args.force_dir,
            velocity_manifest=velocity_manifest,
            force_manifest=force_manifest,
            frame_group=frame_group,
            output_path=output_path,
            panel_width=args.panel_width,
        )
        print(f"[ok] wrote {output_path}")


if __name__ == "__main__":
    main()
