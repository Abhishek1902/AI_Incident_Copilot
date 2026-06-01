"""Seed the local incident_db with a synthetic incident.

Reads app/data/sample_incidents.json and ingests every entry via the same
public ingestion functions used by the API (ingest_logs / ingest_events /
ingest_pipeline_metadata). On-conflict-do-nothing means re-running is safe.

Usage:
    python scripts/seed.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.session import SessionLocal  # noqa: E402
from app.services.ingestion import (  # noqa: E402
    ingest_events,
    ingest_logs,
    ingest_pipeline_metadata,
)

FIXTURE = Path(__file__).parent.parent / "app" / "data" / "sample_incidents.json"


def _parse_occurred_at(items: list[dict]) -> list[dict]:
    for item in items:
        item["occurred_at"] = datetime.fromisoformat(
            item["occurred_at"].replace("Z", "+00:00")
        )
    return items


def main() -> None:
    payload = json.loads(FIXTURE.read_text())
    logs = _parse_occurred_at(payload.get("logs", []))
    events = _parse_occurred_at(payload.get("events", []))
    metadata = _parse_occurred_at(payload.get("metadata", []))

    db = SessionLocal()
    try:
        r_logs = ingest_logs(logs, db)
        r_events = ingest_events(events, db)
        r_meta = ingest_pipeline_metadata(metadata, db)
    finally:
        db.close()

    total_in = r_logs.ingested + r_events.ingested + r_meta.ingested
    total_skip = r_logs.skipped + r_events.skipped + r_meta.skipped
    print(
        f"Seeded {total_in} events "
        f"(logs={r_logs.ingested}, events={r_events.ingested}, metadata={r_meta.ingested}); "
        f"skipped {total_skip} duplicates."
    )


if __name__ == "__main__":
    main()
