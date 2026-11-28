"""The gallery's view of the output index (``files.db``): every generated
image with what the gallery filters and sorts by.

:class:`~remi.files.FileStore` owns the ``files`` table; this module keeps a
side table ``gallery_meta`` (one row per output: the model read from its
infotext, its category — ``txt2img``, ``img2img``, ``extras``, ``grids``,
``saved`` or ``other``, from the job that made it or the folder it is in —
and the star) in the same database, on a connection of its own. Rows are
filled lazily for outputs that have none yet, so files indexed by the
folder scan or by older versions join in on the next query.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import copy
import json
import math
import os
import re
import secrets
import sqlite3
import stat
import threading
import time
import weakref
import zipfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from remi.files import ID_RE, FileRecord, FileStore, close_db, connect, path_inside
from remi.image_meta import clip_infotext
from remi.ui.errors import ParamError, ValidationError
from remi.ui.sentence import Sentence

Sort = Literal["newest", "oldest", "name", "largest"]
SORTS: tuple[Sort, ...] = ("newest", "oldest", "name", "largest")
CATEGORIES = ("txt2img", "img2img", "extras", "grids", "saved", "other")
MAX_QUERY_TERMS = 8
MAX_ZIP_FILES = 2000
EXPORT_MAX_AGE = 3600.0
MAX_SCAN_ROUNDS = 10
MISSING_DIRS_TTL = 10.0
"""seconds :meth:`GalleryIndex._missing_roots` trusts its last look at the
output folders (a drive that comes back shows up within this time)"""
"""pages of rows :meth:`GalleryIndex.list` reads past missing files before it
returns a short page (with a cursor to go on)"""

# Output folder options per category (``modules.shared_options``), most specific first
CATEGORY_FOLDERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("grids", ("outdir_grids", "outdir_txt2img_grids", "outdir_img2img_grids")),
    ("saved", ("outdir_save",)),
    ("extras", ("outdir_extras_samples",)),
    ("img2img", ("outdir_img2img_samples",)),
    ("txt2img", ("outdir_txt2img_samples", "outdir_hires_samples")),
)

_MODEL_RE = re.compile(r'(?:^|,\s*)Model:\s*("(?:\\.|[^"\\])*"|[^,\n]*)')

_SCHEMA = """
CREATE TABLE IF NOT EXISTS gallery_meta (
    file_id TEXT PRIMARY KEY,
    model TEXT,
    category TEXT NOT NULL,
    starred INTEGER NOT NULL DEFAULT 0,
    missing_since REAL
);
CREATE INDEX IF NOT EXISTS gallery_meta_model ON gallery_meta(model);
CREATE TABLE IF NOT EXISTS boards (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    color TEXT,
    created REAL NOT NULL,
    updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS board_items (
    board_id TEXT NOT NULL,
    file_id TEXT NOT NULL,
    added REAL NOT NULL,
    PRIMARY KEY (board_id, file_id)
);
CREATE INDEX IF NOT EXISTS board_items_file ON board_items(file_id);
CREATE INDEX IF NOT EXISTS board_items_board ON board_items(board_id, added DESC);
CREATE TABLE IF NOT EXISTS asset_lineage (
    child_id TEXT NOT NULL,
    parent_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    job_id TEXT,
    chain_id TEXT,
    stage_index INTEGER,
    created REAL NOT NULL,
    PRIMARY KEY (child_id, parent_id, relation)
);
CREATE INDEX IF NOT EXISTS asset_lineage_parent ON asset_lineage(parent_id);
CREATE INDEX IF NOT EXISTS asset_lineage_job ON asset_lineage(job_id);
CREATE INDEX IF NOT EXISTS asset_lineage_chain ON asset_lineage(chain_id, stage_index);
CREATE INDEX IF NOT EXISTS files_kind_name ON files(kind, name);
CREATE INDEX IF NOT EXISTS files_kind_area ON files(kind, COALESCE(width * height, 0));
CREATE INDEX IF NOT EXISTS gallery_meta_starred ON gallery_meta(starred, file_id);
"""
# The last three serve the gallery's sorts by name and by size (the expression is the one
# GalleryIndex.list orders by) and its star filter, which read and sorted every output without
# them. ``IF NOT EXISTS``: a database an older version made gets them when it is next opened.

LINEAGE_RELATIONS = ("init_img", "mask", "upscale_from", "retry_of", "reference", "chain_stage")
MAX_LINEAGE_DEPTH = 8
MAX_LINEAGE_NODES = 128
_IN_CHUNK = 500
"""ids per ``IN (…)`` list (well under SQLite's variable limit)"""


MAX_BOARD_NAME = 120


def _board_name(name: str | None) -> str:
    """A board's name, stripped; a blank or too long one is a refused value on ``name``."""

    cleaned = (name or "").strip()
    if not cleaned:
        raise ValidationError([ParamError("board name is required", param="name")])
    if len(cleaned) > MAX_BOARD_NAME:
        raise ValidationError([ParamError(Sentence("board name must be at most {{maximum}} characters", maximum=MAX_BOARD_NAME), param="name")])
    return cleaned


class BoardError(ValueError):
    """Board operation error carrying an HTTP status code."""

    def __init__(self, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.status = status

_COLUMNS = "f.id, f.kind, f.path, f.name, f.mime, f.size, f.width, f.height, f.created, f.job_id, f.tab, f.infotext"


def model_of(infotext: str | None) -> str | None:
    """The ``Model`` entry of an infotext's parameter line."""

    if not infotext:
        return None
    last = infotext.strip().rsplit("\n", 1)[-1]
    match = _MODEL_RE.search(last)
    if not match:
        return None
    value = match.group(1).strip()
    if value.startswith('"') and value.endswith('"') and len(value) >= 2:
        value = value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    return value[:200] or None


def output_folders() -> list[tuple[str, str]]:
    """``(category, absolute folder)`` of the configured output folders."""

    from remi.setting_values import setting

    try:
        from modules import shared

        opts = shared.opts
    except Exception:
        return []
    found = []
    for category, names in CATEGORY_FOLDERS:
        for name in names:
            value = setting(name, "", opts=opts)  # a list or a number there: unset
            if value:
                found.append((category, os.path.realpath(os.path.abspath(value))))
    return found


def category_of(record_path: str, tab: str | None, folders: Iterable[tuple[str, str]]) -> str:
    path = os.path.realpath(record_path)
    for category, folder in folders:
        if path_inside(path, folder) is not None:
            if category in ("grids", "saved") or tab is None:
                return category
            break
    return tab if tab in ("txt2img", "img2img", "extras") else "other"


def _like(query: str | None) -> list[str]:
    terms = [term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") for term in (query or "").split() if term]
    return [f"%{term}%" for term in terms[:MAX_QUERY_TERMS]]


@dataclass(frozen=True)
class GalleryItem:
    record: FileRecord
    model: str | None
    category: str
    starred: bool


@dataclass(frozen=True)
class GalleryQuery:
    query: str | None = None
    category: str | None = None
    model: str | None = None
    since: float | None = None
    until: float | None = None
    starred: bool | None = None
    sort: Sort = "newest"
    board_id: str | None = None


class GalleryIndex:
    def __init__(self, files: FileStore, *, folders: Callable[[], list[tuple[str, str]]] = output_folders) -> None:
        self._files_ref: weakref.ref[FileStore] = weakref.ref(files)
        self.folders = folders
        self._lock = threading.RLock()
        self._db = connect(files.root / "files.db")
        self._ensure_schema()
        self._finalizer = weakref.finalize(self, close_db, self._db)
        self._synced: int | None = None
        """``PRAGMA data_version`` at the last complete :meth:`sync` (it
        changes when another connection — the file store's — commits)"""
        self._missing: tuple[float, int, tuple[str, ...]] | None = None
        """``(checked at, data_version, roots)`` of :meth:`_missing_roots`"""
        self._facets: tuple[tuple[Any, ...], dict[str, Any]] | None = None
        """``(key, answer)`` of the last :meth:`facets`"""
        self._has_missing: bool = True

    @property
    def files(self) -> FileStore:
        store = self._files_ref()
        if store is None:
            raise RuntimeError("FileStore has been closed")
        return store

    def _ensure_schema(self) -> None:
        with self._lock:
            rows = self._db.execute("SELECT type, name FROM sqlite_master WHERE type IN ('table', 'index')").fetchall()
            tables = {name for kind, name in rows if kind == "table"}
            indexes = {name for kind, name in rows if kind == "index"}
            required_tables = {"gallery_meta", "boards", "board_items", "asset_lineage"}
            required_indexes = {
                "gallery_meta_model",
                "board_items_file",
                "board_items_board",
                "asset_lineage_parent",
                "asset_lineage_job",
                "asset_lineage_chain",
                "files_kind_name",
                "files_kind_area",
                "gallery_meta_starred",
            }
            if not (required_tables <= tables and required_indexes <= indexes):
                self._db.executescript(_SCHEMA)
            cols = {row[1] for row in self._db.execute("PRAGMA table_info(gallery_meta)").fetchall()}
            if "missing_since" not in cols:
                self._db.execute("ALTER TABLE gallery_meta ADD COLUMN missing_since REAL")

    def close(self) -> None:
        with self._lock:
            self._finalizer.detach()
            close_db(self._db)

    @contextlib.contextmanager
    def _transaction(self) -> Iterator[None]:
        """Writes that land together or not at all (the connection autocommits otherwise)."""

        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                if self._db.in_transaction:  # some errors (a full disk) roll back on their own
                    self._db.execute("ROLLBACK")
                raise
            self._db.execute("COMMIT")

    # region Index

    def sync(self, batch: int = 2000, *, force: bool = False) -> int:
        """Fill ``gallery_meta`` for outputs that have no row yet; a no-op
        while the file index is unchanged since the last sync."""

        with self._lock:
            version = self._db.execute("PRAGMA data_version").fetchone()[0]
            if not force and version == self._synced:
                if self._has_missing:
                    self._restore_returned()
                return 0
        folders = self.folders()
        categories: dict[tuple[str, str | None], str] = {}

        def category(path: str, tab: str | None) -> str:
            # once per folder: paths are stored resolved, so a file's category is its folder's
            key = (os.path.dirname(path), tab)
            if key not in categories:
                categories[key] = category_of(path, tab, folders)
            return categories[key]

        added = 0
        while True:
            with self._lock:
                rows = self._db.execute(
                    "SELECT f.id, f.path, f.tab, f.infotext FROM files f LEFT JOIN gallery_meta g ON g.file_id = f.id WHERE f.kind = 'output' AND g.file_id IS NULL LIMIT ?",
                    (batch,),
                ).fetchall()
                if rows:
                    self._db.executemany(
                        "INSERT OR IGNORE INTO gallery_meta (file_id, model, category) VALUES (?, ?, ?)",
                        [(file_id, model_of(infotext), category(path, tab)) for file_id, path, tab, infotext in rows],
                    )
                if len(rows) < batch:
                    self._restore_returned()
                    self._synced = version
                    return added + len(rows)
            added += len(rows)

    def _restore_returned(self) -> None:
        """Clear ``missing_since`` on files that reappeared on disk."""

        with self._lock:
            rows = self._db.execute(
                "SELECT g.file_id, f.path FROM gallery_meta g JOIN files f ON f.id = g.file_id WHERE g.missing_since IS NOT NULL"
            ).fetchall()
            if not rows:
                self._has_missing = False
                return
        restored = [(file_id,) for file_id, path in rows if os.path.isfile(path)]
        if restored:
            with self._transaction():
                self._db.executemany("UPDATE gallery_meta SET missing_since = NULL WHERE file_id = ?", restored)
            if len(restored) == len(rows):
                self._has_missing = False

    def _missing_roots(self) -> tuple[str, ...]:
        """The topmost missing folders that hold indexed outputs (a drive that
        is not mounted, an output folder that was moved away). Their rows stay
        in the index (stars and all, for when they come back) but are left out
        of every count and page, so the totals match what can be shown."""

        now = time.monotonic()
        with self._lock:
            version = self._db.execute("PRAGMA data_version").fetchone()[0]
            cached = self._missing
            if cached is not None and cached[1] == version and now - cached[0] < MISSING_DIRS_TTL:
                return cached[2]
            folders = {os.path.dirname(path) for (path,) in self._db.execute("SELECT DISTINCT path FROM files WHERE kind = 'output'")}
        roots: set[str] = set()
        known: dict[str, bool] = {}

        def exists(folder: str) -> bool:
            if folder not in known:
                known[folder] = os.path.isdir(folder)
            return known[folder]

        for folder in folders:
            if not folder or exists(folder):
                continue
            top = folder
            while True:
                parent = os.path.dirname(top)
                if parent == top or not parent or exists(parent):
                    break
                top = parent
            roots.add(top)
        result = tuple(sorted(roots))
        with self._lock:
            self._missing = (now, version, result)
        return result

    def _available(self) -> tuple[list[str], list[Any]]:
        """SQL conditions leaving out the outputs under :meth:`_missing_roots`."""

        where: list[str] = []
        args: list[Any] = []
        for root in self._missing_roots():
            prefix = root.rstrip(os.sep) + os.sep
            where.append("substr(f.path, 1, ?) <> ?")  # exact (LIKE ignores case)
            args += [len(prefix), prefix]
        return where, args

    def _prune_meta(self) -> None:
        with self._transaction():
            self._db.execute("DELETE FROM gallery_meta WHERE file_id NOT IN (SELECT id FROM files WHERE kind = 'output')")
            self._db.execute("DELETE FROM board_items WHERE file_id NOT IN (SELECT id FROM files WHERE kind = 'output')")
            self._db.execute("DELETE FROM asset_lineage WHERE child_id NOT IN (SELECT id FROM files) OR parent_id NOT IN (SELECT id FROM files)")

    def prune_meta(self) -> None:
        """Remove orphaned gallery metadata, board items, and lineage rows."""

        self._prune_meta()

    def _mark_missing(self, ids: list[str]) -> None:
        """Hide temporarily vanished outputs from gallery queries without
        destroying stars, board memberships, lineage, or thumbnails."""

        now = time.time()
        with self._transaction():
            self._db.executemany(
                "UPDATE gallery_meta SET missing_since = COALESCE(missing_since, ?) WHERE file_id = ?",
                [(now, file_id) for file_id in ids],
            )
            self._has_missing = True

    def prune_missing(self, max_age: float) -> int:
        """Permanently forget outputs that stayed missing longer than ``max_age``
        seconds (while their parent directory still exists)."""

        cutoff = time.time() - max_age
        with self._lock:
            rows = self._db.execute(
                "SELECT g.file_id, f.path FROM gallery_meta g JOIN files f ON f.id = g.file_id WHERE g.missing_since IS NOT NULL AND g.missing_since < ?",
                (cutoff,),
            ).fetchall()
        restored: list[str] = []
        vanished: list[str] = []
        for file_id, path in rows:
            if os.path.isfile(path):
                restored.append(file_id)
            elif os.path.isdir(os.path.dirname(path)):
                vanished.append(file_id)
        if restored:
            with self._transaction():
                self._db.executemany("UPDATE gallery_meta SET missing_since = NULL WHERE file_id = ?", [(fid,) for fid in restored])
        if vanished:
            self._forget(vanished)
        return len(vanished)

    def _forget(self, ids: list[str]) -> None:
        """Drop the index rows (and thumbnails) of deleted files; unlike
        :meth:`FileStore.delete`, the paths are not marked removed, so the
        folder scan finds them again should they come back. The rows go in
        one transaction: none is left pointing at a file row that is gone."""

        with self._transaction():
            self._db.executemany("DELETE FROM files WHERE id = ? AND kind = 'output'", [(file_id,) for file_id in ids])
            self._db.executemany("DELETE FROM gallery_meta WHERE file_id = ?", [(file_id,) for file_id in ids])
            self._db.executemany("DELETE FROM board_items WHERE file_id = ?", [(file_id,) for file_id in ids])
            self._db.executemany("DELETE FROM asset_lineage WHERE child_id = ? OR parent_id = ?", [(file_id, file_id) for file_id in ids])
        for file_id in ids:
            self.files.delete_thumbnails(file_id)

    # endregion

    # region Queries

    def _filters(self, q: GalleryQuery) -> tuple[list[str], list[Any]]:
        available, available_args = self._available()
        where, args = ["f.kind = 'output'", "g.missing_since IS NULL", *available], list(available_args)
        if q.category:
            where.append("g.category = ?")
            args.append(q.category)
        if q.model:
            where.append("g.model = ?")
            args.append(q.model)
        if q.since is not None:
            where.append("f.created >= ?")
            args.append(float(q.since))
        if q.until is not None:
            where.append("f.created < ?")
            args.append(float(q.until))
        if q.starred is not None:
            # starred: the few starred rows through gallery_meta_starred; unstarred (most rows): the
            # unary + keeps that index out, so the page is read in its order as without the filter
            where.append("g.starred = 1" if q.starred else "+g.starred = 0")
        if q.board_id:
            where.append("f.id IN (SELECT file_id FROM board_items WHERE board_id = ?)")
            args.append(q.board_id)
        for pattern in _like(q.query):
            where.append("(f.infotext LIKE ? ESCAPE '\\' OR f.name LIKE ? ESCAPE '\\' OR g.model LIKE ? ESCAPE '\\')")
            args += [pattern] * 3
        return where, args

    def list(self, q: GalleryQuery = GalleryQuery(), *, cursor: str | None = None, limit: int = 100) -> tuple[list[GalleryItem], str | None, int | None]:
        """A page: ``(items, next_cursor, total)``; ``total`` only for the
        first page (no ``cursor``). Outputs whose file cannot be found are
        left out (see the module docstring)."""

        self.sync()
        sort = q.sort if q.sort in SORTS else "newest"
        # created and name are NOT NULL: bare, so files_kind_created_id and files_kind_name serve the order; the
        # size's expression is files_kind_area's (another spelling would sort every output)
        key, direction = {"newest": ("f.created", "DESC"), "oldest": ("f.created", "ASC"), "name": ("f.name", "ASC"), "largest": ("COALESCE(f.width * f.height, 0)", "DESC")}[sort]
        where, args = self._filters(q)
        op = "<" if direction == "DESC" else ">"
        after = _position(cursor, text=sort == "name")
        limit = min(max(int(limit), 1), 200)
        total = self._count(where, args) if cursor is None else None

        items: list[GalleryItem] = []
        missing: list[str] = []
        unavailable = False
        last: tuple[Any, int] | None = None
        more = False
        for _round in range(MAX_SCAN_ROUNDS):
            page_where, page_args = list(where), list(args)
            if after is not None:
                page_where.append(f"({key} {op} ? OR ({key} = ? AND f.rowid {op} ?))")
                page_args += [after[0], after[0], after[1]]
            sql = (
                f"SELECT {_COLUMNS}, g.model, g.category, g.starred, {key}, f.rowid FROM files f JOIN gallery_meta g ON g.file_id = f.id "
                f"WHERE {' AND '.join(page_where)} ORDER BY {key} {direction}, f.rowid {direction} LIMIT ?"
            )
            with self._lock:
                rows = self._db.execute(sql, [*page_args, limit + 1]).fetchall()
            more = False
            for index, row in enumerate(rows):
                if len(items) == limit:
                    more = True  # a row past the last one returned
                    break
                if index == limit:  # the probe row of a batch that came up short: read on from it
                    more = True
                    break
                last = (row[-2], row[-1])
                record = FileRecord(*row[:12])
                try:
                    st = os.stat(record.path)
                except (FileNotFoundError, NotADirectoryError):
                    if os.path.isdir(os.path.dirname(record.path)):
                        missing.append(record.id)
                    else:
                        unavailable = True  # its folder went away since the last look
                except OSError:
                    unavailable = True  # transient I/O or permission error: do not purge metadata
                else:
                    if stat.S_ISREG(st.st_mode):
                        items.append(GalleryItem(record, row[12], row[13], bool(row[14])))
                    else:
                        unavailable = True
            if len(items) == limit or not more:
                break
            after = last
        else:
            more = True  # gave up reading past missing files: the client goes on with the cursor
        if missing:
            self._mark_missing(missing)
        if unavailable:
            with self._lock:
                self._missing = None
        if total is not None and (missing or unavailable):
            # count again, so the total matches the pages ("Images: 10006" over no tiles helps no one)
            where, args = self._filters(q)
            total = self._count(where, args)
        next_cursor = _cursor(*last) if more and last is not None else None
        return items, next_cursor, total

    def _count(self, where: list[str], args: list[Any]) -> int:
        with self._lock:
            return self._db.execute(f"SELECT COUNT(*) FROM files f JOIN gallery_meta g ON g.file_id = f.id WHERE {' AND '.join(where)}", args).fetchone()[0]

    def get(self, file_id: str) -> GalleryItem | None:
        self.sync()
        with self._lock:
            row = self._db.execute(
                f"SELECT {_COLUMNS}, g.model, g.category, g.starred FROM files f JOIN gallery_meta g ON g.file_id = f.id WHERE f.id = ? AND f.kind = 'output'",
                (file_id,),
            ).fetchone()
        if row is None:
            return None
        record = self.files._valid_record(FileRecord(*row[:12]))
        if record is None:
            return None
        return GalleryItem(record, row[12], row[13], bool(row[14]))

    def facets(self) -> dict[str, Any]:
        """Filter values with counts: models, categories, date bounds, stars. Three passes
        over every output, so the answer is kept until the outputs change (another
        connection's commit: ``data_version``; this one's writes: ``total_changes``) or
        the missing folders do."""

        self.sync()
        available, args = self._available()
        where = " AND ".join(["f.kind = 'output'", "g.missing_since IS NULL", *available])
        with self._lock:
            key = (self._db.execute("PRAGMA data_version").fetchone()[0], self._db.total_changes, tuple(args))
            if self._facets is not None and self._facets[0] == key:
                return copy.deepcopy(self._facets[1])
            models = self._db.execute(
                f"SELECT g.model, COUNT(*) FROM files f JOIN gallery_meta g ON g.file_id = f.id WHERE {where} AND g.model IS NOT NULL GROUP BY g.model ORDER BY COUNT(*) DESC, g.model LIMIT 500",
                args,
            ).fetchall()
            categories = dict(self._db.execute(f"SELECT g.category, COUNT(*) FROM files f JOIN gallery_meta g ON g.file_id = f.id WHERE {where} GROUP BY g.category", args).fetchall())
            oldest, newest, total, starred = self._db.execute(f"SELECT MIN(f.created), MAX(f.created), COUNT(*), COALESCE(SUM(g.starred), 0) FROM files f JOIN gallery_meta g ON g.file_id = f.id WHERE {where}", args).fetchone()
            result = {
                "models": [{"name": name, "count": count} for name, count in models],
                "categories": [{"name": name, "count": categories.get(name, 0)} for name in CATEGORIES],
                "oldest": oldest,
                "newest": newest,
                "total": total,
                "starred": starred,
            }
            self._facets = (key, result)
        return copy.deepcopy(result)

    # endregion

    # region Changes

    def kept_ids(self) -> set[str]:
        """Outputs the user chose to keep: starred or on a board (housekeeping spares them)."""

        with self._lock:
            rows = self._db.execute("SELECT file_id FROM gallery_meta WHERE starred = 1 UNION SELECT file_id FROM board_items").fetchall()
        return {file_id for (file_id,) in rows}

    def set_starred(self, ids: list[str], starred: bool) -> int:
        self.sync()
        ids = list(dict.fromkeys(ids))[:MAX_ZIP_FILES]
        if not ids:
            return 0
        with self._lock:
            return self._db.execute(f"UPDATE gallery_meta SET starred = ? WHERE file_id IN ({','.join('?' * len(ids))})", (int(starred), *ids)).rowcount

    def delete(self, ids: list[str], *, delete_files: bool = False) -> int:
        """Remove outputs from the gallery (and from disk with ``delete_files``)."""

        removed = 0
        for file_id in dict.fromkeys(ids):
            record = self.files.get(file_id)
            if record is None or record.kind != "output":
                continue
            if self.files.delete(file_id, delete_file=delete_files):
                removed += 1
        if removed:
            self._prune_meta()
        return removed

    def prune_exports(self, max_age: float = EXPORT_MAX_AGE, exports_dir: Path | None = None) -> int:
        """Delete zips (and leftovers of failed ones) older than ``max_age``
        seconds, and the file index rows of exports that are gone or old."""

        exports = exports_dir or self.files.root / "exports"
        cutoff = time.time() - max_age
        removed = 0
        if exports.is_dir():
            for old in [*exports.glob("*.zip"), *exports.glob("*.tmp")]:
                try:
                    if old.stat().st_mtime < cutoff:
                        old.unlink()
                        removed += 1
                except OSError:
                    pass
        with self._lock:
            rows = self._db.execute("SELECT id, path, created FROM files WHERE kind = 'export'").fetchall()
        inside = os.path.realpath(exports)
        for file_id, path, created in rows:
            if created < cutoff or not os.path.isfile(path):
                self.files.delete(file_id)  # an export: only the row
                if os.path.dirname(os.path.realpath(path)) == inside:  # not a zip the user saved elsewhere ("Save" folder)
                    Path(path).unlink(missing_ok=True)
        return removed

    def export_zip(self, ids: list[str], exports_dir: Path | None = None) -> tuple[Path, int]:
        """Pack the outputs ``ids`` (originals, as stored) into a new zip
        under ``exports/``; files gone from disk are skipped. Exports older
        than an hour are removed first (and by the hourly housekeeping)."""

        exports = exports_dir or self.files.root / "exports"
        exports.mkdir(parents=True, exist_ok=True)
        self.prune_exports(exports_dir=exports)
        records = [record for record in (self.files.get(file_id) for file_id in dict.fromkeys(ids[:MAX_ZIP_FILES])) if record is not None and record.kind == "output"]
        if not records:
            raise ValueError("nothing to pack")
        path = exports / f"gallery-{time.strftime('%Y%m%d-%H%M%S')}-{os.urandom(3).hex()}.zip"
        used: set[str] = set()
        temp = path.with_suffix(".tmp")
        packed = 0
        try:
            with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                for record in records:
                    try:
                        archive.write(record.path, _unique(record.name, used))
                    except (FileNotFoundError, PermissionError, IsADirectoryError):
                        continue  # deleted (or locked) since: raised before the entry is written
                    packed += 1
            if not packed:
                raise ValueError("nothing to pack")
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
        return path, packed

    # endregion

    # region Boards

    def _board_dict(self, row: tuple[Any, ...]) -> dict[str, Any]:
        board_id, name, description, color, created, updated, item_count, cover_file_id = row
        return {
            "id": board_id,
            "name": name,
            "description": description,
            "color": color,
            "item_count": int(item_count or 0),
            "cover_file_id": cover_file_id,
            "created": float(created),
            "updated": float(updated),
        }

    def list_boards(self) -> list[dict[str, Any]]:
        self.sync()
        with self._lock:
            rows = self._db.execute(
                """
                SELECT b.id, b.name, b.description, b.color, b.created, b.updated,
                       (SELECT COUNT(*) FROM board_items bi JOIN files f ON f.id = bi.file_id WHERE bi.board_id = b.id AND f.kind = 'output'),
                       (SELECT bi.file_id FROM board_items bi JOIN files f ON f.id = bi.file_id WHERE bi.board_id = b.id AND f.kind = 'output' ORDER BY bi.added DESC LIMIT 1)
                FROM boards b
                ORDER BY b.updated DESC, b.name ASC
                """
            ).fetchall()
        return [self._board_dict(row) for row in rows]

    def get_board(self, board_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                """
                SELECT b.id, b.name, b.description, b.color, b.created, b.updated,
                       (SELECT COUNT(*) FROM board_items bi JOIN files f ON f.id = bi.file_id WHERE bi.board_id = b.id AND f.kind = 'output'),
                       (SELECT bi.file_id FROM board_items bi JOIN files f ON f.id = bi.file_id WHERE bi.board_id = b.id AND f.kind = 'output' ORDER BY bi.added DESC LIMIT 1)
                FROM boards b
                WHERE b.id = ?
                """,
                (board_id,),
            ).fetchone()
        return self._board_dict(row) if row is not None else None

    def create_board(self, name: str, *, description: str = "", color: str | None = None) -> dict[str, Any]:
        cleaned = _board_name(name)
        board_id = secrets.token_hex(8)
        now = time.time()
        desc = (description or "").strip()[:1000]
        clean_color = color.strip()[:32] if isinstance(color, str) and color.strip() else None
        with self._lock:
            try:
                self._db.execute(
                    "INSERT INTO boards (id, name, description, color, created, updated) VALUES (?, ?, ?, ?, ?, ?)",
                    (board_id, cleaned, desc, clean_color, now, now),
                )
            except sqlite3.IntegrityError as error:
                raise BoardError(f"a board named {cleaned!r} already exists", status=409) from error
        board = self.get_board(board_id)
        assert board is not None
        return board

    def update_board(self, board_id: str, *, name: str | None = None, description: str | None = None, color: Any = ...) -> dict[str, Any]:
        existing = self.get_board(board_id)
        if existing is None:
            raise BoardError(f"no board {board_id!r}", status=404)
        new_name = existing["name"]
        if name is not None:
            new_name = _board_name(name)
        new_desc = existing["description"] if description is None else description.strip()[:1000]
        if color is ...:
            new_color = existing["color"]
        else:
            new_color = color.strip()[:32] if isinstance(color, str) and color.strip() else None
        now = time.time()
        with self._lock:
            try:
                self._db.execute(
                    "UPDATE boards SET name = ?, description = ?, color = ?, updated = ? WHERE id = ?",
                    (new_name, new_desc, new_color, now, board_id),
                )
            except sqlite3.IntegrityError as error:
                raise BoardError(f"a board named {new_name!r} already exists", status=409) from error
        board = self.get_board(board_id)
        assert board is not None
        return board

    def delete_board(self, board_id: str) -> bool:
        with self._transaction():
            self._db.execute("DELETE FROM board_items WHERE board_id = ?", (board_id,))
            return self._db.execute("DELETE FROM boards WHERE id = ?", (board_id,)).rowcount > 0

    def add_to_board(self, board_id: str, file_ids: Iterable[str]) -> int:
        unique_ids = list(dict.fromkeys(fid for fid in file_ids if isinstance(fid, str) and fid))[:MAX_ZIP_FILES]
        now = time.time()
        added = 0
        with self._transaction():
            if self.get_board(board_id) is None:
                raise BoardError(f"no board {board_id!r}", status=404)
            if not unique_ids:
                return 0
            existing: set[str] = set()
            for start in range(0, len(unique_ids), _IN_CHUNK):
                chunk = unique_ids[start : start + _IN_CHUNK]
                existing.update(
                    fid
                    for (fid,) in self._db.execute(
                        f"SELECT id FROM files WHERE kind = 'output' AND id IN ({','.join('?' * len(chunk))})",
                        chunk,
                    ).fetchall()
                )
            for file_id in unique_ids:
                if file_id not in existing:
                    continue
                cur = self._db.execute(
                    "INSERT OR IGNORE INTO board_items (board_id, file_id, added) VALUES (?, ?, ?)",
                    (board_id, file_id, now),
                )
                added += cur.rowcount
            if added:
                self._db.execute("UPDATE boards SET updated = ? WHERE id = ?", (now, board_id))
        return added

    def remove_from_board(self, board_id: str, file_ids: Iterable[str]) -> int:
        unique_ids = list(dict.fromkeys(fid for fid in file_ids if isinstance(fid, str) and fid))[:MAX_ZIP_FILES]
        now = time.time()
        with self._transaction():
            if self.get_board(board_id) is None:
                raise BoardError(f"no board {board_id!r}", status=404)
            if not unique_ids:
                return 0
            removed = self._db.execute(
                f"DELETE FROM board_items WHERE board_id = ? AND file_id IN ({','.join('?' * len(unique_ids))})",
                (board_id, *unique_ids),
            ).rowcount
            if removed:
                self._db.execute("UPDATE boards SET updated = ? WHERE id = ?", (now, board_id))
        return removed

    def boards_for_file(self, file_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                """
                SELECT b.id, b.name, b.description, b.color, b.created, b.updated,
                       (SELECT COUNT(*) FROM board_items bi2 WHERE bi2.board_id = b.id),
                       (SELECT bi3.file_id FROM board_items bi3 WHERE bi3.board_id = b.id ORDER BY bi3.added DESC LIMIT 1)
                FROM boards b
                JOIN board_items bi ON bi.board_id = b.id
                WHERE bi.file_id = ?
                ORDER BY b.name ASC
                """,
                (file_id,),
            ).fetchall()
        return [self._board_dict(row) for row in rows]

    # endregion

    # region Asset Lineage

    def record_lineage(
        self,
        child_id: str,
        parent_id: str,
        relation: str,
        *,
        job_id: str | None = None,
        chain_id: str | None = None,
        stage_index: int | None = None,
    ) -> None:
        if not child_id or not parent_id or child_id == parent_id:
            return
        with self._lock:
            self._db.execute(
                # an edge indexed again keeps its first time (the lists sort by it); an unchanged one is not written
                "INSERT INTO asset_lineage (child_id, parent_id, relation, job_id, chain_id, stage_index, created) VALUES (?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT (child_id, parent_id, relation) DO UPDATE SET job_id = excluded.job_id, chain_id = excluded.chain_id, stage_index = excluded.stage_index"
                " WHERE job_id IS NOT excluded.job_id OR chain_id IS NOT excluded.chain_id OR stage_index IS NOT excluded.stage_index",
                (child_id, parent_id, relation, job_id, chain_id, stage_index, time.time()),
            )

    def index_job_lineage(self, job: Any, output_file_ids: Iterable[str], *, job_store: Any = None) -> int:
        """Index lineage edges linking ``output_file_ids`` to all parent assets of ``job``."""

        parents = extract_job_parents(job, job_store=job_store, files=self.files)
        meta = getattr(job, "meta", None) or {}
        chain_id = meta.get("chain_id") if isinstance(meta.get("chain_id"), str) else None
        stage_index = meta.get("stage_index") if isinstance(meta.get("stage_index"), int) else None
        job_id = getattr(job, "id", None)
        count = 0
        for child_id in output_file_ids:
            if not isinstance(child_id, str) or not child_id:
                continue
            for parent_id, relation in parents:
                if parent_id != child_id:
                    self.record_lineage(child_id, parent_id, relation, job_id=job_id, chain_id=chain_id, stage_index=stage_index)
                    count += 1
        return count

    _NODE_COLUMNS = "f.id, f.path, f.name, f.width, f.height, f.created, f.job_id, f.tab, g.model"
    """a lineage node's file columns (``files f`` joined with ``gallery_meta g``)"""

    @staticmethod
    def _node(
        file_id: str,
        row: tuple | None,
        *,
        relation: str,
        depth: int,
        iso: Callable[[float | None], str | None] | None = None,
        edge_job_id: str | None = None,
        chain_id: str | None = None,
        stage_index: int | None = None,
    ) -> dict[str, Any]:
        """A lineage node from its file columns (:attr:`_NODE_COLUMNS`; None: no such file)."""

        if row is None or row[0] is None:
            return {
                "file_id": file_id,
                "relation": relation,
                "depth": depth,
                "job_id": edge_job_id,
                "chain_id": chain_id,
                "stage_index": stage_index,
                "name": None,
                "width": None,
                "height": None,
                "tab": None,
                "model": None,
                "created_at": None,
                "exists": False,
            }
        fid, path, name, width, height, created, job_id, tab, model = row
        return {
            "file_id": fid,
            "relation": relation,
            "depth": depth,
            "job_id": job_id or edge_job_id,
            "chain_id": chain_id,
            "stage_index": stage_index,
            "name": name,
            "width": width,
            "height": height,
            "tab": tab,
            "model": model,
            "created_at": iso(created) if iso is not None else created,
            "exists": bool(path and os.path.isfile(path)),
        }

    def _edges(self, ids: list[str], *, up: bool) -> dict[str, list[tuple]]:
        """The edges of ``ids`` in one query per chunk: ``{id: [(other id,
        relation, job_id, chain_id, stage_index, *other's file columns)]}``,
        each list oldest first. ``up``: their parents, else their children."""

        mine, other = ("child_id", "parent_id") if up else ("parent_id", "child_id")
        found: dict[str, list[tuple]] = {}
        for start in range(0, len(ids), _IN_CHUNK):
            chunk = ids[start : start + _IN_CHUNK]
            with self._lock:
                rows = self._db.execute(
                    f"SELECT a.{mine}, a.{other}, a.relation, a.job_id, a.chain_id, a.stage_index, a.created, {self._NODE_COLUMNS}"
                    f" FROM asset_lineage a LEFT JOIN files f ON f.id = a.{other} LEFT JOIN gallery_meta g ON g.file_id = a.{other}"
                    f" WHERE a.{mine} IN ({','.join('?' * len(chunk))})",
                    chunk,
                ).fetchall()
            for row in sorted(rows, key=lambda row: row[6]):
                found.setdefault(row[0], []).append((*row[1:6], *row[7:]))
        return found

    def _index_job_of(self, file_id: str, job_id: str | None, job_store: Any) -> int:
        """Index the lineage of the job that made ``file_id``; the number of edges."""

        job = job_store.get(job_id) if job_id and job_store is not None else None
        if job is None:
            return 0
        out_ids = [o["id"] for o in (job.outputs or []) if isinstance(o, dict) and o.get("id")]
        if file_id not in out_ids:
            out_ids.append(file_id)
        return self.index_job_lineage(job, out_ids, job_store=job_store)

    def _walk(
        self,
        file_id: str,
        job_id: str | None,
        *,
        up: bool,
        job_store: Any,
        iso: Callable[[float | None], str | None] | None,
    ) -> tuple[list[dict[str, Any]], list[tuple]]:
        """The ancestors (``up``) or descendants of ``file_id``, breadth first,
        one level at a time, and ``file_id``'s own edges. Going up, a file
        without edges whose job was never indexed (it finished before
        lineage was) is indexed on the way; one with edges is not again, so
        viewing writes nothing."""

        found: list[dict[str, Any]] = []
        visited = {file_id}
        jobs = {file_id: job_id}
        level = [file_id]
        own: list[tuple] = []
        depth = 0
        while level and depth < MAX_LINEAGE_DEPTH and len(found) < MAX_LINEAGE_NODES:
            edges = self._edges(level, up=up)
            if up and job_store is not None:
                unindexed = [node for node in level if node not in edges and jobs.get(node)]
                if sum(self._index_job_of(node, jobs[node], job_store) for node in unindexed):
                    edges.update(self._edges(unindexed, up=True))
            if depth == 0:
                own = edges.get(file_id, [])
            following: list[str] = []
            for node in level:
                if len(found) >= MAX_LINEAGE_NODES:
                    break
                for other, relation, edge_job_id, chain_id, stage_index, *columns in edges.get(node, ()):
                    if other in visited:
                        continue
                    visited.add(other)
                    found.append(self._node(other, tuple(columns), relation=relation, depth=depth + 1, iso=iso, edge_job_id=edge_job_id, chain_id=chain_id, stage_index=stage_index))
                    jobs[other] = columns[6]
                    following.append(other)
            level = following
            depth += 1
        return found, own

    def lineage_for(
        self,
        file_id: str,
        *,
        job_store: Any = None,
        iso: Callable[[float | None], str | None] | None = None,
    ) -> dict[str, Any]:
        """Return ``{file_id, item, ancestors, siblings, descendants}`` for
        ``file_id``: a few queries per level of the walk, not per node."""

        self.sync()
        with self._lock:
            row = self._db.execute(f"SELECT {self._NODE_COLUMNS} FROM files f LEFT JOIN gallery_meta g ON g.file_id = f.id WHERE f.id = ?", (file_id,)).fetchone()
        ancestors, own_edges = self._walk(file_id, row[6] if row else None, up=True, job_store=job_store, iso=iso)
        descendants, _ = self._walk(file_id, None, up=False, job_store=None, iso=iso)
        own_job_id, own_chain_id, own_stage_index = own_edges[0][2:5] if own_edges else (None, None, None)
        item_node = self._node(file_id, row, relation="self", depth=0, iso=iso, edge_job_id=own_job_id, chain_id=own_chain_id, stage_index=own_stage_index)

        # Siblings: same job_id or sharing an immediate parent
        siblings: list[dict[str, Any]] = []
        seen_siblings: set[str] = {file_id}
        effective_job_id = item_node["job_id"]
        if effective_job_id:
            with self._lock:
                same_job_rows = self._db.execute(
                    f"SELECT {self._NODE_COLUMNS} FROM files f LEFT JOIN gallery_meta g ON g.file_id = f.id WHERE f.job_id = ? AND f.kind = 'output' AND f.id != ? ORDER BY f.created ASC",
                    (effective_job_id, file_id),
                ).fetchall()
            for sibling in same_job_rows:
                if sibling[0] not in seen_siblings:
                    seen_siblings.add(sibling[0])
                    siblings.append(self._node(sibling[0], sibling, relation="same_job", depth=0, iso=iso, edge_job_id=effective_job_id))
        parent_ids = list(dict.fromkeys(edge[0] for edge in own_edges))
        sib_rows: list[tuple] = []
        for start in range(0, len(parent_ids), _IN_CHUNK):
            chunk = parent_ids[start : start + _IN_CHUNK]
            with self._lock:
                sib_rows += self._db.execute(
                    f"SELECT DISTINCT a.child_id, a.job_id, a.chain_id, a.stage_index, {self._NODE_COLUMNS}"
                    " FROM asset_lineage a LEFT JOIN files f ON f.id = a.child_id LEFT JOIN gallery_meta g ON g.file_id = a.child_id"
                    f" WHERE a.parent_id IN ({','.join('?' * len(chunk))}) AND a.child_id != ?",
                    (*chunk, file_id),
                ).fetchall()
        for sib_id, edge_job_id, chain_id, stage_index, *columns in sib_rows:
            if sib_id not in seen_siblings and len(siblings) < MAX_LINEAGE_NODES:
                seen_siblings.add(sib_id)
                siblings.append(self._node(sib_id, tuple(columns), relation="same_parent", depth=0, iso=iso, edge_job_id=edge_job_id, chain_id=chain_id, stage_index=stage_index))

        return {
            "file_id": file_id,
            "item": item_node if item_node["exists"] or item_node["name"] is not None else None,
            "ancestors": ancestors,
            "siblings": siblings,
            "descendants": descendants,
        }

    # endregion


def _cursor(value: Any, rowid: int) -> str:
    """A page cursor: the sort value and rowid of the last row read (JSON:
    file names may contain any character)."""

    return base64.urlsafe_b64encode(json.dumps([value, rowid]).encode()).decode().rstrip("=")


def _position(cursor: str | None, *, text: bool) -> tuple[Any, int] | None:
    if not cursor:
        return None
    try:
        value, rowid = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
    except (ValueError, TypeError, binascii.Error):
        return None
    if not isinstance(rowid, int) or isinstance(rowid, bool) or isinstance(value, bool) or not isinstance(value, str if text else (int, float)):
        return None
    if not text and isinstance(value, float) and not math.isfinite(value):
        return None
    if not all(-(2**63) <= number < 2**63 for number in (rowid, value) if isinstance(number, int)):  # SQLite binds 64-bit integers only
        return None
    return value, rowid


def _unique(name: str, used: set[str]) -> str:
    base, ext = os.path.splitext(name or "image")
    candidate, index = base + ext, 1
    while candidate.lower() in used:
        candidate = f"{base}-{index}{ext}"
        index += 1
    used.add(candidate.lower())
    return candidate


_indexes: weakref.WeakKeyDictionary[FileStore, GalleryIndex] = weakref.WeakKeyDictionary()
_indexes_lock = threading.Lock()


def gallery_for(files: FileStore) -> GalleryIndex:
    """One index per file store (process-wide, like the services)."""

    with _indexes_lock:
        index = _indexes.get(files)
        if index is None:
            index = _indexes[files] = GalleryIndex(files)
        return index


def close_gallery(files: FileStore) -> None:
    """Close the index of ``files`` (its own database connection), if one
    was opened: :meth:`remi.server.Services.close`."""

    with _indexes_lock:
        index = _indexes.pop(files, None)
    if index is not None:
        index.close()


def item_json(item: GalleryItem, iso: Callable[[float | None], str | None]) -> dict[str, Any]:
    record = item.record
    return {
        "id": record.id,
        "name": record.name,
        "width": record.width,
        "height": record.height,
        "size": record.size,
        "mime": record.mime,
        "created_at": iso(record.created),
        "job_id": record.job_id,
        "tab": record.tab,
        "category": item.category,
        "model": item.model,
        "starred": item.starred,
        "infotext": record.infotext,
    }


def reveal(path: str) -> None:
    """Show ``path`` in the file manager of the machine running the WebUI
    (selected where the platform can)."""

    import platform
    import subprocess

    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    system = platform.system()
    if system == "Windows":
        subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
    elif system == "Darwin":
        subprocess.Popen(["open", "-R", path])
    elif "microsoft-standard-WSL2" in platform.uname().release:
        windows = subprocess.check_output(["wslpath", "-w", path], text=True).strip()
        subprocess.Popen(["explorer.exe", "/select,", windows])
    else:
        subprocess.Popen(["xdg-open", os.path.dirname(path)])


def board_json(board: dict[str, Any], iso: Callable[[float | None], str | None]) -> dict[str, Any]:
    return {
        "id": board["id"],
        "name": board["name"],
        "description": board["description"],
        "color": board["color"],
        "item_count": board["item_count"],
        "cover_file_id": board["cover_file_id"],
        "created_at": iso(board["created"]) or "",
        "updated_at": iso(board["updated"]) or "",
    }


def _extract_file_ids_from_value(value: Any, *, files: FileStore | None = None) -> list[str]:
    found: list[str] = []

    def walk(item: Any) -> None:
        if isinstance(item, str) and ID_RE.match(item):
            if files is None or files.get(item) is not None:
                found.append(item)
        elif isinstance(item, Mapping):
            for v in item.values():
                walk(v)
        elif isinstance(item, (list, tuple)):
            for elem in item:
                walk(elem)

    walk(value)
    return list(dict.fromkeys(found))


_REFERENCE_SCRIPT_HINT = ("stitch", "reference", "ref", "control", "ipadapter", "image")


def extract_job_parents(job: Any, *, job_store: Any = None, files: FileStore | None = None) -> list[tuple[str, str]]:
    """Extract ``(parent_file_id, relation)`` pairs from a :class:`~remi.jobs.Job`."""

    edges: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(parent_id: Any, relation: str) -> None:
        if isinstance(parent_id, str) and ID_RE.match(parent_id):
            key = (parent_id, relation)
            if key not in seen:
                seen.add(key)
                edges.append(key)

    core = getattr(job, "core", None) or {}
    scripts = getattr(job, "scripts", None) or {}
    meta = getattr(job, "meta", None) or {}

    if isinstance(core, Mapping):
        for key in ("init_img", "image"):
            add(core.get(key), "init_img")
        for key in ("mask", "mask_upload", "sketch"):
            add(core.get(key), "mask")
        for key in ("image_batch", "reference_images"):
            for fid in _extract_file_ids_from_value(core.get(key), files=files):
                add(fid, "reference")

    if isinstance(meta, Mapping):
        add(meta.get("upscale_from"), "upscale_from")
        for fid in _extract_file_ids_from_value(meta.get("upstream_file_ids"), files=files):
            add(fid, "chain_stage")

    retry_of = getattr(job, "retry_of", None)
    if retry_of and job_store is not None:
        parent_job = job_store.get(retry_of)
        if parent_job is not None:
            for out in parent_job.outputs or ():
                if isinstance(out, Mapping) and out.get("kind", "image") == "image":
                    add(out.get("id"), "retry_of")

    if isinstance(scripts, Mapping):
        for script_id, values in scripts.items():
            if not isinstance(values, Mapping):
                continue
            sid_lower = str(script_id).lower()
            is_ref_script = any(hint in sid_lower for hint in _REFERENCE_SCRIPT_HINT)
            for param_name, param_value in values.items():
                pname = str(param_name).lower()
                if is_ref_script or any(tok in pname for tok in ("image", "ref", "stitch", "mask")):
                    for fid in _extract_file_ids_from_value(param_value, files=files):
                        add(fid, "reference")

    return edges


# region Structured Parameter Diff

# Both run on text of any size (an image's own infotext, the compare form) under the GIL, so they
# stay linear. A tag never spans another "<" ("<lora:a<lora:a..." tried every "<" to the end of the
# text; "<lora:sty <lora:style:0.7>" now reads the inner tag). A key starts the line or follows a
# comma or a colon, the only places the first try at a chunk can fail and the next one succeed, and
# has no leading space to give back (a run without a colon was tried at every character); a key of
# only spaces still matches (and is skipped) as before.
_LORA_TAG_RE = re.compile(r"<(?:lora|lyco):([^:<>]+):([^<>]+)>", re.IGNORECASE)
_KV_RE = re.compile(r'(?<![^,:])(\s*+[^\s:,][^:,\n]*?|\s++):\s*+("(?:\\.|[^"\\])*"|\{[^{}]*\}|\[[^\[\]]*\]|[^,\n]*?)(?:,|$)')

_CORE_PARAM_MAP: dict[str, str] = {
    "Steps": "scripts.sampler.steps",
    "Sampler": "scripts.sampler.sampler_name",
    "Schedule type": "scripts.sampler.scheduler",
    "CFG scale": "core.cfg_scale",
    "Distilled CFG Scale": "core.distilled_cfg_scale",
    "Seed": "scripts.seed.seed",
    "Size": "core.size",
    "Model": "core.model",
    "Model hash": "core.model_hash",
    "Denoising strength": "core.denoising_strength",
    "Clip skip": "core.clip_skip",
    "Hires upscale": "core.hr_scale",
    "Hires steps": "core.hr_second_pass_steps",
    "Hires upscaler": "core.hr_upscaler",
    "Hires resize": "core.hr_resize",
    "Hires CFG Scale": "core.hr_cfg",
    "RNG": "core.rng",
    "Module 1": "core.module_1",
    "Version": "core.version",
}

_EXT_GROUP_PREFIXES: tuple[tuple[str, str], ...] = (
    ("ADetailer", "ADetailer"),
    ("ControlNet", "ControlNet"),
    ("DAVE", "DAVE"),
    ("FreeFuse", "FreeFuse"),
    ("Anima", "Anima"),
    ("Regional", "Regional"),
    ("Spectrum", "Spectrum"),
    ("NAG", "NAG"),
    ("Lora", "LoRA"),
    ("TI", "Embeddings"),
    ("Hires", "Hires"),
    ("Soft inpainting", "Soft Inpainting"),
    ("MultiDiffusion", "MultiDiffusion"),
    ("Detail Daemon", "Detail Daemon"),
)


def _unquote(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    return text


def _coerce_scalar(value: str) -> Any:
    text = _unquote(value)
    low = text.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        if "." not in text and "e" not in low:
            return int(text)
    except ValueError:
        pass
    try:
        num = float(text)
        if math.isfinite(num):
            return num
    except ValueError:
        pass
    return text


def _extract_loras(prompt: str, negative: str, params: Mapping[str, str]) -> dict[str, Any]:
    loras: dict[str, Any] = {}
    for text in (prompt, negative):
        for match in _LORA_TAG_RE.finditer(text or ""):
            name = match.group(1).strip()
            raw_weight = match.group(2).strip()
            try:
                parsed = float(raw_weight)
                weight: Any = parsed if math.isfinite(parsed) else raw_weight
            except ValueError:
                weight = raw_weight
            loras[name] = weight
    lora_hashes = params.get("Lora hashes")
    if lora_hashes:
        cleaned = _unquote(lora_hashes)
        for part in cleaned.split(","):
            if ":" in part:
                name = part.split(":", 1)[0].strip()
                if name and name not in loras:
                    loras[name] = 1.0
    return loras


def _split_prompt_tokens(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    tokens: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in text:
        if ch in "([<":
            depth += 1
            buf.append(ch)
        elif ch in ")]>":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch in (",\n") and depth == 0:
            part = "".join(buf).strip()
            if part:
                tokens.append(part)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        tokens.append(tail)
    return tokens


def _diff_tokens(text_a: str, text_b: str) -> dict[str, Any]:
    tokens_a = _split_prompt_tokens(text_a)
    tokens_b = _split_prompt_tokens(text_b)
    counts_a: dict[str, int] = {}
    for tok in tokens_a:
        counts_a[tok] = counts_a.get(tok, 0) + 1
    counts_b: dict[str, int] = {}
    for tok in tokens_b:
        counts_b[tok] = counts_b.get(tok, 0) + 1

    unchanged: list[str] = []
    removed: list[str] = []
    rem_b = dict(counts_b)
    for tok in tokens_a:
        if rem_b.get(tok, 0) > 0:
            unchanged.append(tok)
            rem_b[tok] -= 1
        else:
            removed.append(tok)

    added: list[str] = []
    rem_a = dict(counts_a)
    for tok in tokens_b:
        if rem_a.get(tok, 0) > 0:
            rem_a[tok] -= 1
        else:
            added.append(tok)

    return {
        "a": text_a,
        "b": text_b,
        "added": added,
        "removed": removed,
        "unchanged": unchanged,
    }


def _looks_like_params_line(line: str) -> bool:
    return bool(re.search(r"(?:^|,\s*)(?:Steps|Sampler|CFG scale|Seed|Size|Model|Denoising strength):", line))


def parse_infotext_fields(infotext: str | None) -> dict[str, Any]:
    """Parse A1111/Forge/Remi infotext into prompt, negative, params and LoRAs."""

    # rows stored before infotext was capped on ingest may be longer
    text = clip_infotext(infotext or "").replace("\r\n", "\n").strip()
    if not text:
        return {"prompt": "", "negative_prompt": "", "params": {}, "loras": {}}

    lines = text.split("\n")
    param_line = ""
    if lines and _looks_like_params_line(lines[-1]):
        param_line = lines.pop()

    prompt_lines: list[str] = []
    negative_lines: list[str] = []
    in_negative = False
    for line in lines:
        if not in_negative and line.startswith("Negative prompt:"):
            in_negative = True
            rest = line[len("Negative prompt:") :].strip()
            if rest:
                negative_lines.append(rest)
        elif in_negative:
            negative_lines.append(line)
        else:
            prompt_lines.append(line)

    prompt = "\n".join(prompt_lines).strip()
    negative = "\n".join(negative_lines).strip()
    params: dict[str, str] = {}
    if param_line:
        for match in _KV_RE.finditer(param_line):
            key = match.group(1).strip()
            val = _unquote(match.group(2))
            if key:
                params[key] = val

    loras = _extract_loras(prompt, negative, params)
    return {"prompt": prompt, "negative_prompt": negative, "params": params, "loras": loras}


def _extension_group(key: str) -> str:
    for prefix, group in _EXT_GROUP_PREFIXES:
        if key.lower().startswith(prefix.lower()):
            return group
    return "Other"


def _build_adopt_patch(source: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    """Build a one-click form patch adopting ``source``'s values wherever they differ from ``other``."""

    core: dict[str, Any] = {}
    scripts: dict[str, dict[str, Any]] = {}
    raw_params: dict[str, Any] = {}

    if source["prompt"] != other["prompt"]:
        core["prompt"] = source["prompt"]
    if source["negative_prompt"] != other["negative_prompt"]:
        core["negative_prompt"] = source["negative_prompt"]

    src_p = source["params"]
    oth_p = other["params"]
    for key, val in src_p.items():
        if oth_p.get(key) == val:
            continue
        raw_params[key] = _coerce_scalar(val)
        if key == "Size" and "x" in val.lower():
            parts = val.lower().split("x", 1)
            if parts[0].strip().isdigit() and parts[1].strip().isdigit():
                core["width"] = int(parts[0].strip())
                core["height"] = int(parts[1].strip())
        elif key == "Steps":
            scripts.setdefault("sampler", {})["steps"] = _coerce_scalar(val)
        elif key == "Sampler":
            scripts.setdefault("sampler", {})["sampler_name"] = val
        elif key == "Schedule type":
            scripts.setdefault("sampler", {})["scheduler"] = val
        elif key == "CFG scale":
            core["cfg_scale"] = _coerce_scalar(val)
        elif key == "Distilled CFG Scale":
            core["distilled_cfg_scale"] = _coerce_scalar(val)
        elif key == "Seed":
            scripts.setdefault("seed", {})["seed"] = _coerce_scalar(val)
        elif key == "Denoising strength":
            core["denoising_strength"] = _coerce_scalar(val)
        elif key == "Hires upscale":
            core["enable_hr"] = True
            core["hr_scale"] = _coerce_scalar(val)
        elif key == "Hires steps":
            core["hr_second_pass_steps"] = _coerce_scalar(val)
        elif key == "Hires upscaler":
            core["hr_upscaler"] = val

    return {"core": core, "scripts": scripts, "raw_params": raw_params}


def diff_parameters(infotext_a: str | None, infotext_b: str | None) -> dict[str, Any]:
    """Compute structured diffs between two infotexts and one-click adoption patches."""

    parsed_a = parse_infotext_fields(infotext_a)
    parsed_b = parse_infotext_fields(infotext_b)

    prompt_diff = _diff_tokens(parsed_a["prompt"], parsed_b["prompt"])
    negative_diff = _diff_tokens(parsed_a["negative_prompt"], parsed_b["negative_prompt"])

    params_a: dict[str, str] = parsed_a["params"]
    params_b: dict[str, str] = parsed_b["params"]
    all_keys = list(dict.fromkeys([*params_a.keys(), *params_b.keys()]))

    core_diffs: list[dict[str, Any]] = []
    extension_diffs: list[dict[str, Any]] = []

    for key in all_keys:
        in_a = key in params_a
        in_b = key in params_b
        val_a = _coerce_scalar(params_a[key]) if in_a else None
        val_b = _coerce_scalar(params_b[key]) if in_b else None
        if in_a and in_b and val_a == val_b:
            continue
        kind = "changed" if (in_a and in_b) else ("added" if in_b else "removed")
        if key in _CORE_PARAM_MAP:
            core_diffs.append({"key": key, "field": _CORE_PARAM_MAP[key], "a": val_a, "b": val_b, "kind": kind})
        else:
            extension_diffs.append({"group": _extension_group(key), "key": key, "a": val_a, "b": val_b, "kind": kind})

    loras_a: dict[str, Any] = parsed_a["loras"]
    loras_b: dict[str, Any] = parsed_b["loras"]
    lora_diffs: list[dict[str, Any]] = []
    for name in sorted(set(loras_a) | set(loras_b)):
        in_a, in_b = name in loras_a, name in loras_b
        if in_a and in_b and loras_a[name] == loras_b[name]:
            continue
        kind = "changed" if (in_a and in_b) else ("added" if in_b else "removed")
        lora_diffs.append({"name": name, "a": loras_a.get(name), "b": loras_b.get(name), "kind": kind})

    identical = (
        parsed_a["prompt"] == parsed_b["prompt"]
        and parsed_a["negative_prompt"] == parsed_b["negative_prompt"]
        and not core_diffs
        and not lora_diffs
        and not extension_diffs
    )

    return {
        "identical": identical,
        "prompt_diff": prompt_diff,
        "negative_diff": negative_diff,
        "core_diffs": core_diffs,
        "lora_diffs": lora_diffs,
        "extension_diffs": extension_diffs,
        "adopt_b_patch": _build_adopt_patch(parsed_b, parsed_a),
        "adopt_a_patch": _build_adopt_patch(parsed_a, parsed_b),
    }


def compare_outputs(
    files: FileStore,
    *,
    id_a: str | None = None,
    id_b: str | None = None,
    infotext_a: str | None = None,
    infotext_b: str | None = None,
) -> dict[str, Any]:
    """Compare two gallery outputs (by file id and/or explicit infotext)."""

    text_a = infotext_a
    text_b = infotext_b
    if id_a is not None:
        rec_a = files.get(id_a)
        if rec_a is None:
            raise BoardError(f"no output {id_a!r}", status=404)
        if text_a is None:
            text_a = rec_a.infotext or ""
    if id_b is not None:
        rec_b = files.get(id_b)
        if rec_b is None:
            raise BoardError(f"no output {id_b!r}", status=404)
        if text_b is None:
            text_b = rec_b.infotext or ""
    diff = diff_parameters(text_a, text_b)
    return {
        "id_a": id_a,
        "id_b": id_b,
        "infotext_a": text_a or "",
        "infotext_b": text_b or "",
        **diff,
    }


# endregion


__all__ = [
    "CATEGORIES",
    "LINEAGE_RELATIONS",
    "SORTS",
    "BoardError",
    "GalleryIndex",
    "GalleryItem",
    "GalleryQuery",
    "board_json",
    "category_of",
    "close_gallery",
    "compare_outputs",
    "diff_parameters",
    "extract_job_parents",
    "gallery_for",
    "item_json",
    "model_of",
    "parse_infotext_fields",
    "reveal",
]
