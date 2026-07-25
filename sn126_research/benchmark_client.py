"""Poker44 public training-benchmark client.

Responsibilities:
  * download benchmark chunk-records from the public API
  * follow cursor pagination
  * cache every record on disk keyed by its chunkHash (content-addressed)
  * preserve sourceDate / releaseVersion / split metadata

The public API (no auth required):

    GET /api/v1/benchmark                      -> status (latestSourceDate, ...)
    GET /api/v1/benchmark/releases             -> release history
    GET /api/v1/benchmark/chunks?sourceDate=.. -> chunk-records (paginated)

Terminology (the API overloads the word "chunk"):
  * a *record* is one element of the top-level ``data["chunks"]`` list. It carries
    ``chunkHash``, ``groundTruth`` (list) and ``chunks`` (list of GROUPS).
  * a *group* is one element of ``record["chunks"]`` — one player's hands and the
    single scoring unit the live validator labels. One group == one dataset sample.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

DEFAULT_BASE_URL = "https://api.poker44.net/api/v1/benchmark"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / ".cache"


@dataclass
class ChunkRecord:
    """One benchmark chunk-record (holds several labeled player-groups)."""

    chunk_hash: str
    chunk_id: str
    chunk_index: int
    source_date: str
    release_version: str
    split: str
    ground_truth: List[int]
    ground_truth_labels: List[str]
    groups: List[List[Dict[str, Any]]]  # record["chunks"] : list of hand-lists
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, rec: Dict[str, Any]) -> "ChunkRecord":
        return cls(
            chunk_hash=str(rec.get("chunkHash") or ""),
            chunk_id=str(rec.get("chunkId") or ""),
            chunk_index=int(rec.get("chunkIndex") or 0),
            source_date=str(rec.get("sourceDate") or ""),
            release_version=str(rec.get("releaseVersion") or ""),
            split=str(rec.get("split") or ""),
            ground_truth=list(rec.get("groundTruth") or []),
            ground_truth_labels=list(rec.get("groundTruthLabels") or []),
            groups=list(rec.get("chunks") or []),
            metadata=dict(rec.get("metadata") or {}),
            raw=rec,
        )


class BenchmarkClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        *,
        timeout: int = 90,
        retries: int = 3,
        user_agent: str = "sn126-research/0.1",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.records_dir = self.cache_dir / "records"
        self.manifest_dir = self.cache_dir / "manifests"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.retries = retries
        self.user_agent = user_agent

    # ---- low-level HTTP ----------------------------------------------------
    def _get(self, path: str, **params: Any) -> Any:
        url = f"{self.base_url}{path}"
        query = {k: v for k, v in params.items() if v is not None}
        if query:
            url = f"{url}?{urllib.parse.urlencode(query)}"
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                req = urllib.request.Request(url, headers={"accept": "application/json", "user-agent": self.user_agent})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                if isinstance(payload, dict) and "data" in payload:
                    return payload["data"]
                return payload
            except Exception as exc:  # noqa: BLE001 - transient network errors
                last_exc = exc
                time.sleep(0.5 * attempt)
        raise RuntimeError(f"GET {url} failed after {self.retries} attempts: {last_exc}")

    # ---- discovery ---------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        return self._get("")

    def latest_source_date(self) -> str:
        return str(self.status()["latestSourceDate"])

    def releases(self, *, limit: int = 30, before: Optional[str] = None) -> List[Dict[str, Any]]:
        data = self._get("/releases", limit=limit, before=before)
        if isinstance(data, dict):
            return list(data.get("releases") or data.get("items") or [])
        return list(data or [])

    def recent_source_dates(self, n: int = 3) -> List[str]:
        """Return the n most recent release dates, newest first."""
        dates: List[str] = []
        for rel in self.releases(limit=max(n * 2, 10)):
            d = str(rel.get("sourceDate") or "")
            if d and d not in dates:
                dates.append(d)
            if len(dates) >= n:
                break
        if not dates:
            dates = [self.latest_source_date()]
        return dates

    # ---- cache -------------------------------------------------------------
    def _record_cache_path(self, chunk_hash: str) -> Path:
        return self.records_dir / f"{chunk_hash}.json"

    def _manifest_path(self, source_date: str, split: Optional[str]) -> Path:
        key = f"{source_date}__{split or 'all'}"
        return self.manifest_dir / f"{key}.json"

    def _load_record(self, chunk_hash: str) -> Optional[ChunkRecord]:
        path = self._record_cache_path(chunk_hash)
        if not path.exists():
            return None
        try:
            return ChunkRecord.from_api(json.loads(path.read_text("utf-8")))
        except Exception:
            return None

    def _save_record(self, rec: ChunkRecord) -> None:
        if not rec.chunk_hash:
            return
        self._record_cache_path(rec.chunk_hash).write_text(
            json.dumps(rec.raw, sort_keys=True), "utf-8"
        )

    # ---- main download -----------------------------------------------------
    def iter_chunk_records(
        self,
        source_date: str,
        *,
        split: Optional[str] = None,
        page_limit: int = 24,
        max_records: Optional[int] = None,
        force: bool = False,
    ) -> Iterator[ChunkRecord]:
        """Yield chunk-records for one release date, using cache when possible."""
        manifest_path = self._manifest_path(source_date, split)

        if manifest_path.exists() and not force:
            hashes = json.loads(manifest_path.read_text("utf-8"))
            emitted = 0
            for h in hashes:
                rec = self._load_record(h)
                if rec is None:
                    continue
                yield rec
                emitted += 1
                if max_records is not None and emitted >= max_records:
                    return
            return

        seen_hashes: List[str] = []
        cursor: Optional[str] = None
        emitted = 0
        while True:
            data = self._get(
                "/chunks",
                sourceDate=source_date,
                split=split,
                limit=page_limit,
                cursor=cursor,
            )
            records = data.get("chunks", []) if isinstance(data, dict) else []
            for rec_raw in records:
                rec = ChunkRecord.from_api(rec_raw)
                if not rec.chunk_hash:
                    continue
                self._save_record(rec)
                seen_hashes.append(rec.chunk_hash)
                yield rec
                emitted += 1
                if max_records is not None and emitted >= max_records:
                    manifest_path.write_text(json.dumps(seen_hashes), "utf-8")
                    return
            cursor = data.get("nextCursor") if isinstance(data, dict) else None
            if not cursor:
                break
        manifest_path.write_text(json.dumps(seen_hashes), "utf-8")

    def download(
        self,
        source_date: str,
        *,
        split: Optional[str] = None,
        max_records: Optional[int] = None,
        force: bool = False,
    ) -> List[ChunkRecord]:
        return list(
            self.iter_chunk_records(
                source_date, split=split, max_records=max_records, force=force
            )
        )

    def download_recent(
        self,
        n_dates: int = 3,
        *,
        split: Optional[str] = None,
        max_records_per_date: Optional[int] = None,
        force: bool = False,
    ) -> Dict[str, List[ChunkRecord]]:
        out: Dict[str, List[ChunkRecord]] = {}
        for date in self.recent_source_dates(n_dates):
            out[date] = self.download(
                date, split=split, max_records=max_records_per_date, force=force
            )
        return out


if __name__ == "__main__":  # tiny smoke test
    c = BenchmarkClient()
    s = c.status()
    print("latestSourceDate:", s.get("latestSourceDate"), "release:", s.get("releaseVersion"))
    recs = c.download(c.latest_source_date(), max_records=2)
    print("records:", len(recs), "| groups in rec0:", len(recs[0].groups), "| labels:", recs[0].ground_truth)
