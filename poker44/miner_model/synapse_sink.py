"""Non-blocking persistence for Synapse analyzer records.

A bounded background queue drains to per-date JSONL files. The producer
(forward()) never waits for disk I/O and is never blocked: a full queue DROPS
the record and increments a counter instead of blocking. Analyzer/sink failure
can never crash or alter inference. Only metadata+aggregate records are written
(the analyzer guarantees no raw hands/actions/players/labels).

Env:
  POKER44_SYNAPSE_ENABLED   (default enabled; "0" disables)
  POKER44_SYNAPSE_DIR       (default <repo>/logs/synapse)
  POKER44_SYNAPSE_MAX_QUEUE (default 2000)
  POKER44_SYNAPSE_MAX_BYTES (per-file size rotation, default 50_000_000)
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

_TRUE = {"1", "true", "yes", "on"}
_STOP = object()


class AnalyzerSink:
    def __init__(self, directory: Optional[Path | str] = None, *, enabled: Optional[bool] = None,
                 max_queue: int = 2000, max_bytes: int = 50_000_000) -> None:
        self.enabled = self._resolve_enabled(enabled)
        self.dir = Path(directory or os.getenv("POKER44_SYNAPSE_DIR")
                        or (Path(__file__).resolve().parents[2] / "logs" / "synapse"))
        try:
            self.max_queue = int(os.getenv("POKER44_SYNAPSE_MAX_QUEUE", str(max_queue)))
            self.max_bytes = int(os.getenv("POKER44_SYNAPSE_MAX_BYTES", str(max_bytes)))
        except Exception:
            self.max_queue, self.max_bytes = max_queue, max_bytes
        self.queue: "queue.Queue[Any]" = queue.Queue(maxsize=self.max_queue)
        self.dropped = 0
        self.written = 0
        self._thread: Optional[threading.Thread] = None
        self._closed = False
        if self.enabled:
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
                self._thread = threading.Thread(target=self._consume, daemon=True)
                self._thread.start()
                atexit.register(self.close)
            except Exception:
                self.enabled = False  # cannot start -> silently disable

    @staticmethod
    def _resolve_enabled(enabled: Optional[bool]) -> bool:
        if enabled is not None:
            return bool(enabled)
        raw = os.getenv("POKER44_SYNAPSE_ENABLED")
        if raw is not None:
            return raw.strip().lower() in _TRUE
        return True

    # ---- producer side (must never block/raise) --------------------------
    def submit(self, record: Dict[str, Any]) -> bool:
        """Enqueue a record without blocking. Returns False if dropped/disabled."""
        if not self.enabled:
            return False
        try:
            self.queue.put_nowait(record)
            return True
        except queue.Full:
            self.dropped += 1
            return False
        except Exception:
            self.dropped += 1
            return False

    # ---- consumer side ---------------------------------------------------
    def _file_for(self, record: Dict[str, Any]) -> Path:
        ts = str(record.get("ts_utc") or datetime.now(timezone.utc).isoformat())
        date = ts[:10] if len(ts) >= 10 else datetime.now(timezone.utc).date().isoformat()
        base = self.dir / f"synapse_analyzer_{date}.jsonl"
        try:
            if base.exists() and base.stat().st_size >= self.max_bytes:
                # size sub-rotation within a day
                i = 1
                while (self.dir / f"synapse_analyzer_{date}.jsonl.{i}").exists():
                    i += 1
                base.replace(self.dir / f"synapse_analyzer_{date}.jsonl.{i}")
        except Exception:
            pass
        return base

    def _consume(self) -> None:
        while True:
            item = self.queue.get()
            try:
                if item is _STOP:
                    return
                try:
                    line = json.dumps(item, separators=(",", ":"), ensure_ascii=True, default=str)
                    with open(self._file_for(item), "a", encoding="utf-8") as fh:
                        fh.write(line + "\n")
                    self.written += 1
                except Exception:
                    pass  # never let a bad record kill the consumer
            finally:
                self.queue.task_done()

    def flush(self, timeout: float = 5.0) -> None:
        if not self.enabled:
            return
        try:
            self.queue.join()
        except Exception:
            pass

    def close(self) -> None:
        if self._closed or not self.enabled:
            return
        self._closed = True
        try:
            self.queue.put_nowait(_STOP)
        except Exception:
            pass
        t = self._thread
        if t is not None:
            t.join(timeout=5.0)

    def stats(self) -> Dict[str, int]:
        return {"written": self.written, "dropped": self.dropped,
                "queue_size": self.queue.qsize() if self.enabled else 0}
