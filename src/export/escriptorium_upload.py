"""
Upload corrections to eScriptorium as a new transcription version.

Uses the escriptorium-connector library to create a new transcription
version on each page with the LLM-corrected text. The expert can then
compare the original HTR version with the corrected version line by
line within the eScriptorium interface.

Workflow:
    1. Connect to the eScriptorium instance
    2. For each corrected page:
       a. Load the production JSON
       b. Map line_id -> esc_id to identify original lines
       c. Create or update the corrected transcription
    3. Log progress

Requirements:
    pip install escriptorium-connector

Usage:
    python -m src.export.escriptorium_upload --config config.yaml --input data/corrected

Config (config.yaml):
    escriptorium:
        url: "https://escriptorium.example.com"
        username: "your_username"
        password: "your_password"
        document_id: 123
        transcription_name: "LLM_corrected"
"""

import argparse
import json
import logging
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/escriptorium_upload.log"),
    ],
)
logger = logging.getLogger(__name__)


def connect_escriptorium(config):
    """
    Connect to an eScriptorium instance.

    Args:
        config: dict with escriptorium section (url, username, password).

    Returns:
        EscriptoriumConnector instance.
    """
    try:
        from escriptorium_connector import EscriptoriumConnector
    except ImportError:
        logger.error(
            "escriptorium-connector not installed. "
            "Run: pip install escriptorium-connector"
        )
        raise

    esc_config = config["escriptorium"]

    if not esc_config.get("url"):
        logger.error("eScriptorium URL not configured in config.yaml")
        raise ValueError("escriptorium.url must be set in config.yaml")

    conn = EscriptoriumConnector(
        url=esc_config["url"],
        username=esc_config["username"],
        password=esc_config["password"],
    )

    logger.info(f"Connected to eScriptorium: {esc_config['url']}")
    return conn


def upload_page_corrections(conn, corrected_data, document_id, transcription_name):
    """
    Upload corrections for a single page to eScriptorium.

    Uses the esc_id field from the production JSON to map each corrected
    line back to the original line in eScriptorium.

    Args:
        conn: EscriptoriumConnector instance.
        corrected_data: dict from production JSON.
        document_id: eScriptorium document ID.
        transcription_name: name for the new transcription (e.g. "LLM_corrected").

    Returns:
        True if successful, False on error.
    """
    page_name = corrected_data["page"]
    lines = corrected_data["lines"]

    try:
        # Find the document part matching this page
        parts = conn.get_document_parts(document_id)

        target_part = None
        for part in parts:
            if page_name in str(part.get("filename", "")):
                target_part = part
                break

        if not target_part:
            logger.warning(
                f"Page {page_name} not found in document {document_id}. "
                f"Available pages: {[p.get('filename') for p in parts[:5]]}..."
            )
            return False

        part_id = target_part["pk"]

        # Create or find the target transcription
        transcriptions = conn.get_document_transcriptions(document_id)
        target_transcription = None

        for t in transcriptions:
            if t.get("name") == transcription_name:
                target_transcription = t
                break

        if not target_transcription:
            target_transcription = conn.create_transcription(
                document_id, transcription_name
            )
            logger.info(f"Created new transcription: {transcription_name}")

        transcription_id = target_transcription["pk"]

        # Upload corrected text line by line using esc_id mapping
        for line in lines:
            esc_id = line.get("esc_id", "")
            corrected_text = line["text_corrected"]

            if not esc_id:
                logger.warning(f"Line {line['line_id']} has no esc_id, skipping")
                continue

            try:
                conn.update_part_transcription_line(
                    part_id=part_id,
                    transcription_id=transcription_id,
                    line_id=esc_id,
                    content=corrected_text,
                )
            except Exception as e:
                logger.warning(f"Error updating line {esc_id}: {e}")

        logger.info(f"Uploaded {len(lines)} lines for {page_name}")
        return True

    except Exception as e:
        logger.error(f"Error uploading {page_name}: {e}")
        return False


def upload_all(config_path="config.yaml", input_dir="data/corrected"):
    """Upload all corrected pages to eScriptorium."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    input_dir = Path(input_dir)
    json_files = sorted(input_dir.glob("*.json"))

    if not json_files:
        logger.error(f"No JSON files found in {input_dir}")
        return

    esc_config = config.get("escriptorium", {})
    document_id = esc_config.get("document_id")
    transcription_name = esc_config.get("transcription_name", "LLM_corrected")

    if not document_id:
        logger.error("escriptorium.document_id must be set in config.yaml")
        return

    conn = connect_escriptorium(config)

    success = 0
    errors = 0

    for json_file in json_files:
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Uploading {data['page']}...")

        if upload_page_corrections(conn, data, document_id, transcription_name):
            success += 1
        else:
            errors += 1

    logger.info(f"\nUpload complete: {success} OK, {errors} errors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload corrections to eScriptorium")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", default="data/corrected")
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    upload_all(args.config, args.input)
