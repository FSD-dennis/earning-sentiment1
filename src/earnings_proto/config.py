from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ProjectConfig:
    """Runtime configuration for the local prototype."""

    root_dir: Path = Path(__file__).resolve().parents[2]
    tickers: list[str] = field(
        default_factory=lambda: ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META"]
    )
    events_per_ticker: int = 4
    pre_days: int = 20
    post_days: int = 10
    target_horizons: list[int] = field(default_factory=lambda: [1, 3, 7])

    @property
    def data_raw_dir(self) -> Path:
        return self.root_dir / "data" / "raw"

    @property
    def data_processed_dir(self) -> Path:
        return self.root_dir / "data" / "processed"

    @property
    def reports_dir(self) -> Path:
        return self.root_dir / "reports"

    def ensure_dirs(self) -> None:
        self.data_raw_dir.mkdir(parents=True, exist_ok=True)
        self.data_processed_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
