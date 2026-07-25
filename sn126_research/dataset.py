"""Dataset assembly: benchmark records -> (X, y) with date-based splits.

Rules enforced here:
  * one GROUP (one player) = one sample; label = groundTruth for that group
  * every hand is projected through the official live sanitizer before features
  * splits are by sourceDate (release), NEVER a random row split — bot families
    rotate per release, so random splits leak future families into training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .benchmark_client import BenchmarkClient, ChunkRecord
from .features import FEATURE_NAMES, feature_vector
from .sanitizer import to_live_group


@dataclass
class Sample:
    x: List[float]
    y: int
    source_date: str
    release_version: str
    chunk_id: str
    group_index: int
    n_hands: int


@dataclass
class Dataset:
    samples: List[Sample] = field(default_factory=list)
    feature_names: Tuple[str, ...] = FEATURE_NAMES

    def __len__(self) -> int:
        return len(self.samples)

    def source_dates(self) -> List[str]:
        return sorted({s.source_date for s in self.samples})

    def matrix(self, samples: Optional[Sequence[Sample]] = None) -> Tuple[np.ndarray, np.ndarray]:
        rows = list(self.samples if samples is None else samples)
        X = np.asarray([s.x for s in rows], dtype=float)
        y = np.asarray([s.y for s in rows], dtype=int)
        return X, y

    def by_dates(self, dates: Sequence[str]) -> List[Sample]:
        want = set(dates)
        return [s for s in self.samples if s.source_date in want]

    def split_by_date(
        self, holdout_dates: Sequence[str]
    ) -> Tuple[List[Sample], List[Sample]]:
        want = set(holdout_dates)
        train = [s for s in self.samples if s.source_date not in want]
        val = [s for s in self.samples if s.source_date in want]
        return train, val


def _sample_from_group(
    rec: ChunkRecord, group_index: int, group: List[Dict[str, Any]], label: int, scope: str
) -> Optional[Sample]:
    live = to_live_group(group)
    if not live:
        return None
    return Sample(
        x=feature_vector(live, scope=scope),
        y=int(label),
        source_date=rec.source_date,
        release_version=rec.release_version,
        chunk_id=rec.chunk_id,
        group_index=group_index,
        n_hands=len(group),
    )


def build_dataset_from_records(records: Sequence[ChunkRecord], *, scope: str = "all") -> Dataset:
    ds = Dataset()
    for rec in records:
        labels = rec.ground_truth
        for gi, group in enumerate(rec.groups):
            if gi >= len(labels):
                continue
            sample = _sample_from_group(rec, gi, group, labels[gi], scope)
            if sample is not None:
                ds.samples.append(sample)
    return ds


def build_dataset(
    *,
    n_dates: int = 3,
    max_records_per_date: Optional[int] = None,
    split: Optional[str] = None,
    scope: str = "all",
    client: Optional[BenchmarkClient] = None,
    force: bool = False,
) -> Dataset:
    """Download recent releases and assemble a date-labeled dataset.

    scope="all" aggregates table-wide behavior (recommended). scope="hero"
    restricts to the hero seat (ablation only; near chance on this benchmark).
    """
    client = client or BenchmarkClient()
    by_date = client.download_recent(
        n_dates=n_dates,
        split=split,
        max_records_per_date=max_records_per_date,
        force=force,
    )
    all_records: List[ChunkRecord] = []
    for date in sorted(by_date):
        all_records.extend(by_date[date])
    return build_dataset_from_records(all_records, scope=scope)
