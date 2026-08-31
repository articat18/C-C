"""Capture live research evidence as inert, immutable campaign artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from experiment_engine.checkpoints import _atomic_json_write
from experiment_boundary import resolve_editable_path


MAX_SOURCE_BYTES = 128 * 1024


class ResearchEvidenceError(ValueError):
    """Raised when live source evidence is unsafe or unavailable."""


def capture_source(url: str, *, title: str | None = None, summary: str | None = None) -> dict[str, Any]:
    """Fetch a public HTTPS source and persist a non-executable evidence record."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ResearchEvidenceError("research sources must use a public https URL")
    request = Request(url, headers={"User-Agent": "AutoML-Phase6-Research/1.0"})
    try:
        with urlopen(request, timeout=15) as response:  # nosec B310: HTTPS-only validated above.
            raw = response.read(MAX_SOURCE_BYTES + 1)
    except OSError as exc:
        raise ResearchEvidenceError(f"could not retrieve source: {exc}") from exc
    if len(raw) > MAX_SOURCE_BYTES:
        raise ResearchEvidenceError("research source exceeds the 128 KiB evidence limit")
    text = _plain_text(raw.decode("utf-8", errors="replace"))
    observed_title = title.strip() if title else _title(text)
    if not observed_title:
        observed_title = parsed.netloc
    digest = hashlib.sha256(raw).hexdigest()
    source_id = f"S-{digest[:16]}"
    record = {
        "source_id": source_id,
        "url": url,
        "title": observed_title,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "content_sha256": digest,
        "summary": summary.strip() if summary else text[:2000],
        "content_excerpt": text[:8000],
        "untrusted": True,
        "execution_policy": "reference_only",
    }
    destination = resolve_editable_path(Path("runs") / "research" / f"{source_id}.json")
    if destination.exists():
        return json.loads(destination.read_text(encoding="utf-8"))
    _atomic_json_write(destination, record)
    return record


def search_crossref(query: str, *, rows: int = 5) -> list[dict[str, Any]]:
    """Discover citable academic work through Crossref, then persist each source."""
    if not query.strip() or not 1 <= rows <= 10:
        raise ResearchEvidenceError("query must be non-empty and rows must be 1..10")
    endpoint = "https://api.crossref.org/works?" + urlencode({"query": query, "rows": rows})
    request = Request(endpoint, headers={"User-Agent": "AutoML-Phase6-Research/1.0"})
    try:
        with urlopen(request, timeout=15) as response:  # nosec B310: fixed HTTPS endpoint.
            payload = json.loads(response.read(MAX_SOURCE_BYTES).decode("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchEvidenceError(f"Crossref search failed: {exc}") from exc
    records = []
    for item in payload.get("message", {}).get("items", []):
        url = item.get("URL")
        titles = item.get("title") or []
        if not isinstance(url, str) or not url.startswith("https://"):
            continue
        title = titles[0] if titles and isinstance(titles[0], str) else None
        try:
            records.append(capture_source(url, title=title, summary=f"Crossref result for: {query}"))
        except ResearchEvidenceError:
            # DOI landing pages commonly block automated fetches.  The live
            # Crossref response is still citable evidence and is safer than
            # retrying around publisher access controls.
            records.append(_capture_crossref_metadata(item, query=query))
    return records


def available_sources() -> list[dict[str, Any]]:
    root = resolve_editable_path(Path("runs") / "research")
    if not root.exists():
        return []
    output = []
    for path in sorted(root.glob("S-*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and isinstance(value.get("source_id"), str):
            output.append({key: value.get(key) for key in ("source_id", "url", "title", "content_sha256", "summary")})
    return output


def _capture_crossref_metadata(item: dict[str, Any], *, query: str) -> dict[str, Any]:
    url = str(item["URL"])
    titles = item.get("title") or []
    title = titles[0] if titles and isinstance(titles[0], str) else url
    encoded = json.dumps(item, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    source_id = f"S-{digest[:16]}"
    record = {
        "source_id": source_id,
        "url": url,
        "title": title,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "content_sha256": digest,
        "summary": f"Crossref result for: {query}",
        "content_excerpt": json.dumps(item, sort_keys=True)[:8000],
        "untrusted": True,
        "execution_policy": "reference_only",
        "retrieval_mode": "crossref_metadata",
    }
    destination = resolve_editable_path(Path("runs") / "research" / f"{source_id}.json")
    if destination.exists():
        return json.loads(destination.read_text(encoding="utf-8"))
    _atomic_json_write(destination, record)
    return record


def validate_source_ids(source_ids: list[str] | tuple[str, ...]) -> list[dict[str, Any]]:
    known = {item["source_id"]: item for item in available_sources()}
    if not source_ids:
        raise ResearchEvidenceError("Phase 6 proposals require at least one research source ID")
    missing = [item for item in source_ids if item not in known]
    if missing:
        raise ResearchEvidenceError("unknown research source IDs: " + ", ".join(missing))
    return [known[item] for item in source_ids]


def _plain_text(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", value)
    return " ".join(html.unescape(without_tags).split())


def _title(value: str) -> str | None:
    match = re.search(r"<title[^>]*>(.*?)</title>", value, flags=re.IGNORECASE | re.DOTALL)
    return _plain_text(match.group(1)) if match else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url")
    source.add_argument("--query")
    parser.add_argument("--title")
    parser.add_argument("--summary")
    parser.add_argument("--rows", type=int, default=5)
    args = parser.parse_args()
    from experiment_engine.campaign import configure_campaign
    configure_campaign(args.campaign)
    output = search_crossref(args.query, rows=args.rows) if args.query else [capture_source(args.url, title=args.title, summary=args.summary)]
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
