from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path
try:
    from zoneinfo import ZoneInfo
except ImportError:  # Python 3.8 compatibility.
    ZoneInfo = None

from .io_utils import read_json
from .models import DataSnapshot, MatchRecord, SourceRecord, utc_now_iso


class DataSourceError(RuntimeError):
    pass


class BaseDataSource:
    name = "base"

    def load_matches(self, date: str, timezone_name: str) -> DataSnapshot:
        raise NotImplementedError


class SampleDataSource(BaseDataSource):
    name = "sample"

    def __init__(self, sample_dir: Path | None = None):
        self.sample_dir = sample_dir or Path(__file__).resolve().parent / "sample_data"

    def load_matches(self, date: str, timezone_name: str) -> DataSnapshot:
        fixture = self.sample_dir / f"matches_{date}.json"
        date_specific_fixture = fixture.exists()
        if not date_specific_fixture:
            fixture = self.sample_dir / "matches_default.json"
        if not fixture.exists():
            raise DataSourceError(f"Sample fixture not found in {self.sample_dir}")

        payload = read_json(fixture)
        source_payloads = payload.get("sources", [])
        sources = [SourceRecord.from_dict(item) for item in source_payloads]
        for source in sources:
            source.fetched_at = utc_now_iso()
            if source.url == "__fixture__":
                source.url = str(fixture)

        matches = []
        for index, item in enumerate(payload.get("matches", []), start=1):
            adjusted = dict(item)
            adjusted["date"] = date
            adjusted["timezone"] = timezone_name
            adjusted["kickoff_local"] = self._adjust_kickoff(
                str(item.get("kickoff_local") or f"{date}T20:00:00"),
                date,
                timezone_name,
                index,
                preserve_raw_date=date_specific_fixture,
            )
            if "match_id" not in adjusted:
                home = str(adjusted["home_team"])[:3].upper()
                away = str(adjusted["away_team"])[:3].upper()
                adjusted["match_id"] = f"{date.replace('-', '')}-{index:02d}-{home}-{away}"
            matches.append(MatchRecord.from_dict(adjusted))

        return DataSnapshot(matches=matches, sources=sources)

    @staticmethod
    def _adjust_kickoff(raw: str, date: str, timezone_name: str, index: int, preserve_raw_date: bool = False) -> str:
        try:
            parsed = datetime.fromisoformat(raw)
            if preserve_raw_date:
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=_timezone_for(timezone_name))
                return parsed.isoformat()
            hour = parsed.hour
            minute = parsed.minute
        except ValueError:
            hour = 18 + index * 2
            minute = 0
        local = datetime.fromisoformat(f"{date}T{hour:02d}:{minute:02d}:00")
        return local.replace(tzinfo=_timezone_for(timezone_name)).isoformat()


class LocalJsonDataSource(BaseDataSource):
    name = "local-json"

    def __init__(self, path: Path):
        self.path = path

    def load_matches(self, date: str, timezone_name: str) -> DataSnapshot:
        payload = read_json(self.path)
        sources = [SourceRecord.from_dict(item) for item in payload.get("sources", [])]
        if not sources:
            sources = [
                SourceRecord(
                    source_id="local_json",
                    name="Local JSON match file",
                    kind="local-json",
                    url=str(self.path),
                )
            ]
        matches = []
        for item in payload.get("matches", []):
            if str(item.get("date")) == date:
                item = dict(item)
                item.setdefault("timezone", timezone_name)
                matches.append(MatchRecord.from_dict(item))
        return DataSnapshot(matches=matches, sources=sources)


def get_data_source(name: str, source_json: Path | None = None) -> BaseDataSource:
    normalized = name.strip().lower()
    if source_json is not None:
        return LocalJsonDataSource(source_json)
    if normalized in {"sample", "mock"}:
        return SampleDataSource()
    raise DataSourceError(f"Unknown data source: {name}")


def _timezone_for(timezone_name: str):
    if ZoneInfo is not None:
        return ZoneInfo(timezone_name)
    offsets = {
        "Asia/Shanghai": timezone(timedelta(hours=8)),
        "UTC": timezone.utc,
    }
    if timezone_name not in offsets:
        raise DataSourceError(
            f"Timezone {timezone_name!r} requires Python 3.9+ zoneinfo; use Asia/Shanghai or UTC on Python 3.8."
        )
    return offsets[timezone_name]
