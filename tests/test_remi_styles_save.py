"""Pinning tests for ``modules/styles.py``: atomic ``save_styles``,
thread safety, and resilient ``load_from_csv`` handling."""

from __future__ import annotations

import csv
import errno
import threading

import pytest

from modules.styles import PromptStyle, StyleDatabase


def test_save_styles_is_atomic_on_write_error(tmp_path, monkeypatch):
    csv_path = tmp_path / "styles.csv"
    db = StyleDatabase([csv_path])
    db.styles["One"] = PromptStyle("One", "sunny day", "rain", str(csv_path))
    db.styles["Two"] = PromptStyle("Two", "starry night", "clouds", str(csv_path))
    db.save_styles()
    before = csv_path.read_bytes()

    db.styles["Three"] = PromptStyle("Three", "forest", "fog", str(csv_path))

    real_writer = csv.DictWriter

    class _FailingWriter(real_writer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._rows = 0

        def writerow(self, rowdict):
            self._rows += 1
            if self._rows >= 2:
                raise OSError(errno.ENOSPC, "No space left on device")
            return super().writerow(rowdict)

    monkeypatch.setattr(csv, "DictWriter", _FailingWriter)
    with pytest.raises(OSError) as exc:
        db.save_styles()
    assert exc.value.errno == errno.ENOSPC
    assert csv_path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []


def test_load_from_csv_skips_rows_missing_name(tmp_path):
    csv_path = tmp_path / "styles.csv"
    # Malformed rows: missing name, empty name, and valid rows
    csv_path.write_text(
        "name,prompt,negative_prompt\n"
        ",empty name,bad\n"
        "#commented,ignored,ignored\n"
        "ValidStyle,portrait,blurry\n",
        encoding="utf-8-sig",
        newline="",
    )
    # Also test a CSV where the 'name' column is absent on a short row (csv.DictReader produces None)
    short_csv = tmp_path / "short.csv"
    short_csv.write_text(
        "prompt,negative_prompt,name\n"
        "no-name-col\n"
        "good prompt,good neg,KeepMe\n",
        encoding="utf-8-sig",
        newline="",
    )

    db = StyleDatabase([csv_path, short_csv])
    assert "ValidStyle" in db.styles
    assert db.styles["ValidStyle"].prompt == "portrait"
    assert "KeepMe" in db.styles
    assert db.styles["KeepMe"].prompt == "good prompt"


def test_save_styles_and_reload_are_thread_safe(tmp_path):
    csv_path = tmp_path / "styles.csv"
    db = StyleDatabase([csv_path])
    for idx in range(10):
        db.styles[f"S{idx}"] = PromptStyle(f"S{idx}", f"prompt {idx}", "", str(csv_path))
    db.save_styles()

    errors: list[BaseException] = []

    def saver():
        try:
            for _ in range(15):
                db.save_styles()
        except BaseException as exc:
            errors.append(exc)

    def reloader():
        try:
            for _ in range(15):
                db.reload()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=saver), threading.Thread(target=reloader)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert errors == []
    assert len(db.styles) == 10
