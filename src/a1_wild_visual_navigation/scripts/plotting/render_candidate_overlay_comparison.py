#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_ORIGINAL_DIR = Path(
    "/home/rain/github_upload/Result/velocity_default(with_frame_output_detail)/wvn_frames"
)
DEFAULT_CANDIDATE_DIR = Path("/home/rain/github_upload/Result/velocity_candidate_sigmoid_frames")
DEFAULT_OUTPUT = Path("/home/rain/github_upload/Result/rendered_figures/velocity_candidate_overlay_comparison.png")
DEFAULT_FRAMES = "420,700,1000,1520,2040,2420,2700,2820,3340"
DEFAULT_TITLE = "Original vs Candidate Traversability Overlays"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render original vs candidate traversability overlay comparison sheet."
    )
    parser.add_argument("--original-dir", type=Path, default=DEFAULT_ORIGINAL_DIR)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", default=DEFAULT_FRAMES)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    return parser.parse_args()


def parse_frames(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def load_manifest_rows(frames_dir):
    manifest_path = frames_dir / "manifest.csv"
    with manifest_path.open(newline="") as handle:
        return {int(row["frame_index"]): row for row in csv.DictReader(handle)}


def parse_optional_float(value):
    if value in ("", None):
        return None
    return float(value)


def format_tau(value):
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def find_font(size, bold=False):
    candidates = (
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        )
        if bold
        else (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        )
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def text_size(font, text):
    left, top, right, bottom = font.getbbox(text)
    return right - left, bottom - top


def image_for_row(frames_dir, manifest_row, filename):
    path = frames_dir / manifest_row["frame_dir"] / filename
    return Image.open(path).convert("RGB")


def build_sheet(original_dir, candidate_dir, frame_ids, output_path, title):
    original_rows = load_manifest_rows(original_dir)
    candidate_rows = load_manifest_rows(candidate_dir)

    sample_frame = frame_ids[0]
    sample_image = image_for_row(original_dir, original_rows[sample_frame], "raw.png")
    cell_w, cell_h = sample_image.size

    top_margin = 74
    header_h = 56
    footer_h = 30
    row_gap = 24
    col_gap = 12
    row_labels = ["Raw image", "Original overlay", "Candidate overlay"]

    title_font = find_font(26)
    header_font = find_font(18, bold=True)
    label_font = find_font(18)
    small_font = find_font(15)

    max_label_w = max(text_size(label_font, label)[0] for label in row_labels)
    left_margin = max(170, max_label_w + 36)

    width = left_margin + len(frame_ids) * cell_w + (len(frame_ids) - 1) * col_gap + 36
    height = top_margin + len(row_labels) * cell_h + (len(row_labels) - 1) * row_gap + header_h + footer_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text((width // 2, 20), title, fill="black", anchor="mm", font=title_font)

    for col, frame_id in enumerate(frame_ids):
        x = left_margin + col * (cell_w + col_gap)
        original_tau = parse_optional_float(original_rows[frame_id].get("instant_traversability"))
        candidate_tau = parse_optional_float(candidate_rows[frame_id].get("instant_traversability"))
        draw.text((x + cell_w / 2, top_margin), f"Frame {frame_id}", fill="black", anchor="mm", font=header_font)
        draw.text(
            (x + cell_w / 2, top_margin + 24),
            f"Torig={format_tau(original_tau)}  Tcand={format_tau(candidate_tau)}",
            fill="#444444",
            anchor="mm",
            font=small_font,
        )

        raw_img = image_for_row(original_dir, original_rows[frame_id], "raw.png")
        orig_overlay = image_for_row(original_dir, original_rows[frame_id], "traversability_overlay.png")
        cand_overlay = image_for_row(candidate_dir, candidate_rows[frame_id], "traversability_overlay.png")
        images = [raw_img, orig_overlay, cand_overlay]

        for row_idx, image in enumerate(images):
            y = top_margin + header_h + row_idx * (cell_h + row_gap)
            canvas.paste(image, (x, y))
            draw.rectangle((x, y, x + cell_w, y + cell_h), outline="#d0d0d0", width=1)

    for row_idx, label in enumerate(row_labels):
        y = top_margin + header_h + row_idx * (cell_h + row_gap) + cell_h / 2
        draw.text((left_margin - 18, y), label, fill="black", anchor="rm", font=label_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main():
    args = parse_args()
    frame_ids = parse_frames(args.frames)
    build_sheet(args.original_dir, args.candidate_dir, frame_ids, args.output, args.title)
    print(f"[ok] wrote {args.output}")


if __name__ == "__main__":
    main()
