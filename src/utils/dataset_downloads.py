"""Guards for datasets whose upstream downloader always re-extracts archives."""

from __future__ import annotations

from pathlib import Path

from torch_timeseries.dataset import SolarEnergy, Weather


def _has_expected_line_count(path: Path, expected_lines: int) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("rb") as data_file:
        return sum(1 for _ in data_file) == expected_lines


def _make_download_idempotent(dataset_class, relative_path: str, expected_lines: int) -> None:
    if getattr(dataset_class, "_nsdiff_download_guard", False):
        return

    original_download = dataset_class.download

    def download_if_needed(self):
        output_path = Path(self.dir) / relative_path
        if _has_expected_line_count(output_path, expected_lines):
            return
        original_download(self)
        if not _has_expected_line_count(output_path, expected_lines):
            raise RuntimeError(
                f"Dataset preparation did not produce a complete file: {output_path}"
            )

    dataset_class.download = download_if_needed
    dataset_class._nsdiff_download_guard = True


def configure_idempotent_downloads() -> None:
    """Avoid concurrent archive extraction for the shared Weather/Solar files."""
    _make_download_idempotent(SolarEnergy, "solar_AL.txt", expected_lines=52560)
    _make_download_idempotent(Weather, "weather/weather.csv", expected_lines=52697)
