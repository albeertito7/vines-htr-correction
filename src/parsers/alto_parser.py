"""
ALTO XML parser for eScriptorium HTR output.

Parses ALTO XML files (v2, v3, v4) exported from eScriptorium and extracts
structured data: text content and bounding box coordinates for each text line.

This is the first step of the pipeline. The HTR transcriptions from
eScriptorium are exported as ALTO XML files containing the recognised text
line by line with coordinates within the scanned page image. The parser
reads these XMLs and produces a clean Python structure with the text and
coordinates for each line, needed to:
  - send the text to the LLM for post-correction
  - crop each line image for multimodal approaches

ALTO XML structure:
    - Hierarchy: Page > PrintSpace > TextBlock > TextLine > String
    - Each TextLine contains one or more String elements with CONTENT attribute
    - There are two formats for the TextLine coordinates:
        - Bounding box: HPOS, VPOS, WIDTH, HEIGHT attributes on TextLine
        - Detailed polygon: Shape > Polygon with POINTS attribute
    To crop the images, should we use the bounding box (easier).

Text conventions in the Viñes corpus:
    - ¶  paragraph start marker
    - ¬  end-of-line hyphen (word split across lines, e.g. "Bag¬" + "nères")
    - &quot; represents quotes inside XML attributes

Line types (TAGREFS):
    - LT8  = DefaultLine (normal diary text)
    - LT10 = HeadingLine (titles, dates, headings as "¶Enero de 1915.")
    - LT7  = CustomLine
    - LT9  = DropCapitalLine
    - LT11 = InterlinearLine
    - LT12 = MusicLine

Block types (TextBlock TAGREFS):
    - BT12 = MainZone (main text area)
    - BT13 = MarginTextZone
    - BT15 = NumberingZone
    - Others: CustomZone, DamageZone, GraphicZone, MusicZone, etc.

Usage:
    python alto_parser.py <file.xml or directory>
"""

from lxml import etree
from pathlib import Path


# ALTO namespace URIs (auto-detected from file)
ALTO_NAMESPACES = {
    "alto2": "http://www.loc.gov/standards/alto/ns-v2#",
    "alto3": "http://www.loc.gov/standards/alto/ns-v3#",
    "alto4": "http://www.loc.gov/standards/alto/ns-v4#",
}


def detect_namespace(root):
    """Auto-detect the ALTO version from the root element's namespace."""
    tag = root.tag
    if "{" in tag:
        return tag.split("}")[0].strip("{")
    return None


def parse_alto_file(filepath):
    """
    Parse an ALTO XML file and return structured data.

    Args:
        filepath: path to the ALTO XML file.

    Returns:
        dict with:
            - filename: file stem (without extension)
            - lines: list of dicts with {line_id, esc_id, text, coords}
                where line_id is sequential (line_000, line_001, ...)
                and esc_id is the original eScriptorium ID.
    """
    tree = etree.parse(str(filepath))
    root = tree.getroot()
    ns = detect_namespace(root)

    if ns:
        nsmap = {"a": ns}
        text_lines = root.findall(".//a:TextLine", nsmap)
    else:
        text_lines = root.findall(".//TextLine")

    lines = []
    for i, text_line in enumerate(text_lines):
        # extract lines coordinates
        coords = {
            "x": int(float(text_line.get("HPOS", 0))),
            "y": int(float(text_line.get("VPOS", 0))),
            "w": int(float(text_line.get("WIDTH", 0))),
            "h": int(float(text_line.get("HEIGHT", 0))),
        }

        # extract text: concat all String
        if ns:
            strings = text_line.findall(".//a:String", nsmap)
        else:
            strings = text_line.findall(".//String")

        words = [s.get("CONTENT", "") for s in strings]
        text = " ".join(words)

        # Custom sequential ID + ID eScriptorium
        esc_id = text_line.get("ID", f"unknown_{i}")
        line_id = f"line_{i:03d}"

        lines.append({
            "line_id": line_id,
            "esc_id": esc_id,
            "text": text,
            "coords": coords,
        })

    return {
        "filename": Path(filepath).stem,
        "lines": lines,
    }


def parse_alto_directory(directory):
    """
    Parse all ALTO XML files in a directory.

    Args:
        directory: path to the directory.

    Returns:
        list of dicts (one per file), sorted by filename.
    """
    directory = Path(directory)
    results = []

    for filepath in sorted(directory.glob("*.xml")):
        try:
            result = parse_alto_file(filepath)
            results.append(result)
        except Exception as e:
            print(f"Error parsing {filepath.name}: {e}")

    return results


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python alto_parser.py <file.xml or directory>")
        sys.exit(1)

    path = Path(sys.argv[1])

    if path.is_file():
        result = parse_alto_file(path)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif path.is_dir():
        results = parse_alto_directory(path)
        for r in results:
            print(f"\n--- {r['filename']} ({len(r['lines'])} lines) ---")
            for line in r['lines'][:5]:
                print(f"  [{line['line_id']}] {line['text']}")
            if len(r['lines']) > 5:
                print(f"  ... i {len(r['lines']) - 5} more lines")
    else:
        print(f"Not found: {path}")
