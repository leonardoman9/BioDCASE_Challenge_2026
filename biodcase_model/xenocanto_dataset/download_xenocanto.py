#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_URL = "https://xeno-canto.org/api/3/recordings"
DEFAULT_CUTOFF = "2025-04-01"
SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Species:
    class_name: str
    scientific_name: str
    genus: str
    species: str

    @property
    def slug(self) -> str:
        return slugify(self.class_name)

    @property
    def query(self) -> str:
        return f"gen:{self.genus} sp:{self.species}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_dir)
    species_file = Path(args.species_file)
    species_list = load_species(species_file)
    if args.species:
        requested = {name.lower() for name in args.species}
        species_list = [
            item
            for item in species_list
            if item.class_name.lower() in requested
            or item.scientific_name.lower() in requested
            or item.slug in requested
        ]
        if not species_list:
            raise SystemExit(f"No species matched: {', '.join(args.species)}")

    api_key = args.api_key or os.environ.get("XC_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing Xeno-canto API key. Set XC_API_KEY or pass --api-key. "
            "Create a key from your xeno-canto account page."
        )

    cutoff = parse_iso_date(args.max_uploaded_date) if args.max_uploaded_date else None
    root.mkdir(parents=True, exist_ok=True)
    (root / "raw_audio").mkdir(exist_ok=True)
    (root / "metadata" / "pages").mkdir(parents=True, exist_ok=True)
    manifest_path = root / "metadata" / "recordings.jsonl"
    summary_path = root / "metadata" / "summary.json"

    summary: dict[str, Any] = {
        "api_url": API_URL,
        "max_uploaded_date": args.max_uploaded_date,
        "download_audio": not args.metadata_only,
        "species": {},
    }

    manifest_mode = "a" if args.append_manifest else "w"
    with manifest_path.open(manifest_mode, encoding="utf-8") as manifest:
        for species in species_list:
            species_summary = download_species(species, args, api_key, cutoff, root, manifest)
            summary["species"][species.class_name] = species_summary

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote summary: {summary_path}")
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Xeno-canto metadata/audio for BioDCASE target bird species."
    )
    parser.add_argument("--api-key", default=None, help="Xeno-canto API key. Prefer XC_API_KEY env var.")
    parser.add_argument("--species-file", default=str(SCRIPT_DIR / "species.json"), help="Species JSON file.")
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR), help="Output directory for raw_audio/ and metadata/.")
    parser.add_argument(
        "--species",
        action="append",
        help="Restrict to one class/scientific name/slug. Can be passed multiple times.",
    )
    parser.add_argument(
        "--max-uploaded-date",
        default=DEFAULT_CUTOFF,
        help="Keep only recordings uploaded on or before this ISO date. Use empty string to disable.",
    )
    parser.add_argument(
        "--require-uploaded-date",
        action="store_true",
        help="Skip records where the API response does not expose an uploaded-date field.",
    )
    parser.add_argument("--quality", action="append", help="Add Xeno-canto quality filter, e.g. A or B.")
    parser.add_argument("--type", action="append", help="Add sound type filter, e.g. song or call.")
    parser.add_argument("--country", action="append", help="Add country filter, e.g. Germany.")
    parser.add_argument("--extra-query", action="append", default=[], help="Extra Xeno-canto query token.")
    parser.add_argument("--max-pages", type=int, default=None, help="Limit pages per species for testing.")
    parser.add_argument("--limit-per-species", type=int, default=None, help="Limit accepted records per species.")
    parser.add_argument("--per-page", type=int, default=None, help="Optional API per_page parameter.")
    parser.add_argument("--metadata-only", action="store_true", help="Fetch metadata but do not download audio.")
    parser.add_argument("--append-manifest", action="store_true", help="Append to metadata/recordings.jsonl.")
    parser.add_argument("--overwrite", action="store_true", help="Re-download existing audio files.")
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between requests/downloads.")
    parser.add_argument("--timeout", type=float, default=60.0, help="Request timeout in seconds.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per HTTP request.")
    return parser.parse_args(argv)


def load_species(path: Path) -> list[Species]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Species(**item) for item in data]


def download_species(
    species: Species,
    args: argparse.Namespace,
    api_key: str,
    cutoff: date | None,
    root: Path,
    manifest,
) -> dict[str, Any]:
    print(f"\n== {species.class_name} ({species.scientific_name}) ==")
    audio_dir = root / "raw_audio" / species.slug
    audio_dir.mkdir(parents=True, exist_ok=True)
    page_dir = root / "metadata" / "pages"

    accepted = 0
    skipped_after_cutoff = 0
    skipped_missing_uploaded = 0
    failed_downloads = 0
    page = 1
    num_pages = None

    while True:
        if args.max_pages is not None and page > args.max_pages:
            break
        if args.limit_per_species is not None and accepted >= args.limit_per_species:
            break

        payload = fetch_json(build_api_url(species, args, api_key, page), args.timeout, args.retries)
        page_path = page_dir / f"{species.slug}_page_{page:04d}.json"
        page_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        if "error" in payload:
            raise RuntimeError(f"Xeno-canto API error for {species.class_name}: {payload}")

        num_pages = int(payload.get("numPages") or payload.get("num_pages") or 1)
        recordings = payload.get("recordings") or []
        print(f"page {page}/{num_pages}: {len(recordings)} records")

        for record in recordings:
            uploaded = uploaded_date(record)
            cutoff_status = "ok"
            if cutoff is not None:
                if uploaded is None:
                    cutoff_status = "missing_uploaded"
                    if args.require_uploaded_date:
                        skipped_missing_uploaded += 1
                        continue
                elif uploaded > cutoff:
                    skipped_after_cutoff += 1
                    continue

            audio_path = None
            download_status = "metadata_only"
            if not args.metadata_only:
                try:
                    audio_path = download_audio(record, species, audio_dir, args)
                    download_status = "downloaded"
                except Exception as exc:  # noqa: BLE001 - manifest should capture failed external downloads.
                    failed_downloads += 1
                    download_status = f"failed: {exc}"

            manifest_record = {
                "class_name": species.class_name,
                "scientific_name": species.scientific_name,
                "query": species.query,
                "cutoff_status": cutoff_status,
                "download_status": download_status,
                "local_audio_path": str(audio_path) if audio_path else None,
                "recording": record,
            }
            manifest.write(json.dumps(manifest_record, sort_keys=True) + "\n")
            manifest.flush()
            accepted += 1

            if args.limit_per_species is not None and accepted >= args.limit_per_species:
                break
            time.sleep(args.sleep)

        if page >= num_pages:
            break
        page += 1
        time.sleep(args.sleep)

    return {
        "scientific_name": species.scientific_name,
        "pages_seen": page,
        "num_pages": num_pages,
        "accepted_records": accepted,
        "skipped_after_cutoff": skipped_after_cutoff,
        "skipped_missing_uploaded": skipped_missing_uploaded,
        "failed_downloads": failed_downloads,
    }


def build_api_url(species: Species, args: argparse.Namespace, api_key: str, page: int) -> str:
    query_parts = [species.query]
    for quality in args.quality or []:
        query_parts.append(f"q:{quality}")
    for sound_type in args.type or []:
        query_parts.append(f'type:"{sound_type}"')
    for country in args.country or []:
        query_parts.append(f'cnt:"{country}"')
    query_parts.extend(args.extra_query or [])

    params: dict[str, Any] = {"query": " ".join(query_parts), "page": page, "key": api_key}
    if args.per_page is not None:
        params["per_page"] = args.per_page
    return f"{API_URL}?{urlencode(params)}"


def fetch_json(url: str, timeout: float, retries: int) -> dict[str, Any]:
    data = fetch_bytes(url, timeout, retries)
    return json.loads(data.decode("utf-8"))


def download_audio(record: dict[str, Any], species: Species, audio_dir: Path, args: argparse.Namespace) -> Path:
    recording_id = str(record.get("id") or record.get("xc_id") or "unknown")
    source_url = audio_url(record)
    extension = audio_extension(record, source_url)
    output_path = audio_dir / f"XC{recording_id}_{species.slug}{extension}"
    if output_path.exists() and not args.overwrite:
        return output_path

    content = fetch_bytes(source_url, args.timeout, args.retries)
    output_path.write_bytes(content)
    return output_path


def fetch_bytes(url: str, timeout: float, retries: int) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers={"User-Agent": "BioDCASE-2026-XC-downloader/0.1"})
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2.0 * attempt, 10.0))
    raise RuntimeError(f"failed after {retries} attempts: {url}: {last_error}")


def uploaded_date(record: dict[str, Any]) -> date | None:
    for key in ("uploaded", "upload_date", "date_uploaded", "created"):
        value = record.get(key)
        parsed = parse_maybe_date(value)
        if parsed is not None:
            return parsed
    return None


def audio_url(record: dict[str, Any]) -> str:
    for key in ("file", "download", "audio", "audio_url", "audioUrl"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return normalize_url(value)
    recording_id = record.get("id") or record.get("xc_id")
    if recording_id:
        return f"https://xeno-canto.org/{recording_id}/download"
    raise ValueError("record has no audio URL or id")


def audio_extension(record: dict[str, Any], source_url: str) -> str:
    for key in ("file-name", "file_name", "filename"):
        value = record.get(key)
        if isinstance(value, str):
            suffix = Path(value).suffix.lower()
            if suffix:
                return suffix
    suffix = Path(source_url.split("?", 1)[0]).suffix.lower()
    return suffix if suffix in {".mp3", ".wav", ".flac", ".ogg", ".m4a"} else ".mp3"


def normalize_url(value: str) -> str:
    if value.startswith("//"):
        return "https:" + value
    if value.startswith("/"):
        return "https://xeno-canto.org" + value
    return value


def parse_iso_date(value: str) -> date:
    return date.fromisoformat(value)


def parse_maybe_date(value: Any) -> date | None:
    if not isinstance(value, str) or not value:
        return None
    match = re.search(r"\d{4}-\d{2}-\d{2}", value)
    if not match:
        return None
    try:
        return date.fromisoformat(match.group(0))
    except ValueError:
        return None


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    return slug.strip("_")


if __name__ == "__main__":
    raise SystemExit(main())
