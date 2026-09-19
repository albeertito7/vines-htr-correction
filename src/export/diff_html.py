"""
HTML diff report generator for expert review.

For each corrected page, generates an HTML file showing the original
HTR text alongside the LLM-corrected text, with differences highlighted:
    - Green: text added or corrected by the LLM
    - Red: original text removed or changed
    - No highlight: unchanged text

The expert can visually review each line and decide whether to accept
or reject each correction.

Also generates an index.html linking to all page reports.

Usage:
    python -m src.export.diff_html --input data/corrected --output data/reports
"""

import argparse
import json
import difflib
from datetime import datetime
from pathlib import Path


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Review: {page_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        .stats {{
            background: #fff;
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .stats span {{
            margin-right: 20px;
            font-size: 14px;
        }}
        .stat-label {{
            color: #666;
        }}
        .stat-value {{
            font-weight: bold;
        }}
        .stat-changed {{ color: #e67e22; }}
        .stat-unchanged {{ color: #27ae60; }}
        .line {{
            background: #fff;
            margin-bottom: 8px;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .line-header {{
            background: #ecf0f1;
            padding: 6px 15px;
            font-size: 12px;
            color: #7f8c8d;
            display: flex;
            justify-content: space-between;
        }}
        .line-changed .line-header {{
            background: #ffeaa7;
        }}
        .line-content {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0;
        }}
        .line-original, .line-corrected {{
            padding: 10px 15px;
            font-size: 15px;
            line-height: 1.6;
            font-family: 'Georgia', serif;
        }}
        .line-original {{
            background: #fff5f5;
            border-right: 1px solid #eee;
        }}
        .line-corrected {{
            background: #f0fff0;
        }}
        .line-unchanged .line-original,
        .line-unchanged .line-corrected {{
            background: #fff;
        }}
        .diff-add {{
            background: #a8e6a3;
            padding: 1px 3px;
            border-radius: 3px;
        }}
        .diff-del {{
            background: #f5a6a6;
            padding: 1px 3px;
            border-radius: 3px;
            text-decoration: line-through;
        }}
        .col-header {{
            text-align: center;
            font-size: 12px;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #666;
            padding: 8px;
            background: #fff;
        }}
        .legend {{
            font-size: 13px;
            color: #666;
            margin-bottom: 20px;
        }}
        .legend span {{
            margin-right: 15px;
        }}
    </style>
</head>
<body>
    <h1>Review: {page_name}</h1>
    <div class="stats">
        <span><span class="stat-label">Model:</span> <span class="stat-value">{model}</span></span>
        <span><span class="stat-label">Strategy:</span> <span class="stat-value">{strategy}</span></span>
        <span><span class="stat-label">Lines:</span> <span class="stat-value">{total_lines}</span></span>
        <span class="stat-changed"><span class="stat-label">Changed:</span> <span class="stat-value">{changed_lines}</span></span>
        <span class="stat-unchanged"><span class="stat-label">Unchanged:</span> <span class="stat-value">{unchanged_lines}</span></span>
        <span><span class="stat-label">Date:</span> <span class="stat-value">{timestamp}</span></span>
    </div>
    <div class="legend">
        <span><span class="diff-del">red</span> = original text changed</span>
        <span><span class="diff-add">green</span> = proposed correction</span>
    </div>
    <div class="line-content" style="display: grid; grid-template-columns: 1fr 1fr; margin-bottom: 10px;">
        <div class="col-header">HTR Original</div>
        <div class="col-header">LLM Correction</div>
    </div>
    {lines_html}
</body>
</html>"""


def word_diff(original, corrected):
    """
    Generate HTML with word-level differences highlighted.

    Returns:
        tuple (original_html, corrected_html) with diff spans.
    """
    orig_words = original.split()
    corr_words = corrected.split()

    matcher = difflib.SequenceMatcher(None, orig_words, corr_words)

    orig_html_parts = []
    corr_html_parts = []

    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            orig_html_parts.append(" ".join(orig_words[i1:i2]))
            corr_html_parts.append(" ".join(corr_words[j1:j2]))
        elif op == "replace":
            orig_html_parts.append(
                '<span class="diff-del">' + " ".join(orig_words[i1:i2]) + "</span>"
            )
            corr_html_parts.append(
                '<span class="diff-add">' + " ".join(corr_words[j1:j2]) + "</span>"
            )
        elif op == "delete":
            orig_html_parts.append(
                '<span class="diff-del">' + " ".join(orig_words[i1:i2]) + "</span>"
            )
        elif op == "insert":
            corr_html_parts.append(
                '<span class="diff-add">' + " ".join(corr_words[j1:j2]) + "</span>"
            )

    return " ".join(orig_html_parts), " ".join(corr_html_parts)


def generate_page_report(corrected_data, output_path):
    """
    Generate an HTML diff report for a single corrected page.

    Args:
        corrected_data: dict loaded from production JSON with keys
                        {page, model, strategy, timestamp, lines}.
        output_path: path to save the HTML report.
    """
    page_lines = corrected_data["lines"]
    changed = [l for l in page_lines if l["text_original"] != l["text_corrected"]]
    unchanged = [l for l in page_lines if l["text_original"] == l["text_corrected"]]

    lines_html_parts = []

    for line in page_lines:
        is_changed = line["text_original"] != line["text_corrected"]
        css_class = "line line-changed" if is_changed else "line line-unchanged"

        if is_changed:
            orig_html, corr_html = word_diff(line["text_original"], line["text_corrected"])
        else:
            orig_html = line["text_original"]
            corr_html = line["text_corrected"]

        label = "CHANGED" if is_changed else "OK"

        lines_html_parts.append(f"""
    <div class="{css_class}">
        <div class="line-header">
            <span>{line['line_id']}</span>
            <span>{label}</span>
        </div>
        <div class="line-content">
            <div class="line-original">{orig_html}</div>
            <div class="line-corrected">{corr_html}</div>
        </div>
    </div>""")

    html = HTML_TEMPLATE.format(
        page_name=corrected_data["page"],
        model=corrected_data.get("model", "unknown"),
        strategy=corrected_data.get("strategy", "unknown"),
        total_lines=len(page_lines),
        changed_lines=len(changed),
        unchanged_lines=len(unchanged),
        timestamp=corrected_data.get("timestamp", datetime.now().isoformat()),
        lines_html="\n".join(lines_html_parts),
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Report generated: {output_path} ({len(changed)} changes in {len(page_lines)} lines)")


def generate_all_reports(input_dir, output_dir):
    """
    Generate HTML reports for all corrected pages.

    Args:
        input_dir: directory with production JSONs (data/corrected/).
        output_dir: directory to save HTML reports (data/reports/).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    json_files = sorted(input_dir.glob("**/*.json"))

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    # Skip checkpoint files
    json_files = [f for f in json_files if f.stem != "checkpoint"]

    print(f"Generating reports for {len(json_files)} pages...")

    for json_file in json_files:
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        # Preserve year subdirectory structure
        relative = json_file.relative_to(input_dir)
        output_path = output_dir / relative.with_suffix(".html")
        generate_page_report(data, output_path)

    generate_index(output_dir, input_dir)


def generate_index(output_dir, input_dir):
    """Generate an index.html linking to all page reports."""
    output_dir = Path(output_dir)
    html_files = sorted(output_dir.glob("**/*.html"))
    html_files = [f for f in html_files if f.name != "index.html"]

    links = []
    for html_file in html_files:
        json_file = input_dir / html_file.relative_to(output_dir).with_suffix(".json")
        changed = 0
        total = 0
        if json_file.exists():
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            page_lines = data["lines"]
            total = len(page_lines)
            changed = sum(1 for l in page_lines if l["text_original"] != l["text_corrected"])

        relative_path = html_file.relative_to(output_dir)
        page_name = html_file.stem
        links.append(
            f'<li><a href="{relative_path}">{page_name}</a> '
            f'— {total} lines, {changed} changes</li>'
        )

    index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>HTR Review Reports Index</title>
    <style>
        body {{ font-family: system-ui, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        li {{ margin: 8px 0; }}
        a {{ color: #2980b9; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <h1>HTR Review Reports</h1>
    <p>{len(html_files)} pages processed</p>
    <ul>{"".join(links)}</ul>
</body>
</html>"""

    with open(output_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(index_html)

    print(f"Index generated: {output_dir / 'index.html'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HTML diff reports for review")
    parser.add_argument("--input", default="data/corrected", help="Directory with production JSONs")
    parser.add_argument("--output", default="data/reports", help="Directory for HTML reports")
    args = parser.parse_args()

    generate_all_reports(args.input, args.output)
