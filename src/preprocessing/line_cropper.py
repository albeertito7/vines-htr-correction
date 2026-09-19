"""
Line image cropper for multimodal HTR post-correction.

Crops individual text lines from scanned page images using the bounding
box coordinates (HPOS, VPOS, WIDTH, HEIGHT) extracted by the ALTO XML
parser.

The cropped line images are used in the multimodal correction strategy:
sending both the HTR text and the line image to the LLM so it can visually verify the recognition.

Uses the bounding box rather than the detailed Shape polygon, as it is
sufficient for cropping and much simpler to handle.

An optional padding parameter adds extra margin around the bounding box to ensure no letter
strokes are cut off.

Usage:
    python line_cropper.py <image.png> <file.xml> [output_dir] [padding]
"""

from PIL import Image
from pathlib import Path


def crop_lines(image_path, lines, output_dir, padding=10):
    """
    Crop each text line from the page image using its coordinates.

    Args:
        image_path: path to the original scanned page image.
        lines: list of dicts with {line_id, text, coords: {x, y, w, h}}
               (output from the ALTO parser).
        output_dir: directory to save the cropped line images.
        padding: extra pixels around the bounding box (default: 10).

    Returns:
        list of dicts with {line_id, text, image_path}.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(image_path)
    img_width, img_height = img.size

    results = []

    for line in lines:
        coords = line["coords"]

        # Bounding box with padding, clamped to image boundaries
        x1 = max(0, coords["x"] - padding)
        y1 = max(0, coords["y"] - padding)
        x2 = min(img_width, coords["x"] + coords["w"] + padding)
        y2 = min(img_height, coords["y"] + coords["h"] + padding)

        # Crop and save
        line_img = img.crop((x1, y1, x2, y2))
        output_path = output_dir / f"{line['line_id']}.png"
        line_img.save(output_path)

        results.append({
            "line_id": line["line_id"],
            "text": line["text"],
            "image_path": str(output_path),
        })

    return results


if __name__ == "__main__":
    import sys
    import json

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.parsers.alto_parser import parse_alto_file

    if len(sys.argv) < 3:
        print("Usage: python line_cropper.py <image.png> <file.xml> [output_dir] [padding]")
        sys.exit(1)

    image_path = sys.argv[1]
    xml_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "data/line_images"
    padding = int(sys.argv[4]) if len(sys.argv) > 4 else 10

    # Parse XML
    parsed = parse_alto_file(xml_path)

    # Crop lines
    results = crop_lines(image_path, parsed["lines"], output_dir, padding)

    print(f"Cropped {len(results)} lines from {Path(image_path).name}:")
    for r in results:
        print(f"  [{r['line_id']}] {r['text'][:50]}... -> {r['image_path']}")
