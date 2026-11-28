"""Files the WebUI serves by opaque id only: uploads, generation outputs and
extra-network previews (``/api/v2/files/{id}``, ``/api/v2/thumbs/{id}``).

The browser never sees a filesystem path: every file is registered in a
small SQLite index (``files.db`` under the Remi data directory) and
addressed by an id. Uploads are sniffed by content (not by the declared type
or name), size-limited, deduplicated by SHA-256 and stored under
``uploads/``. Outputs keep their infotext for the gallery search.
"""

from __future__ import annotations

import base64
import contextlib
import dataclasses
import hashlib
import logging
import math
import os
import re
import secrets
import sqlite3
import threading
import time
import weakref
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, PngImagePlugin

from remi.image_meta import clip_infotext, utf8_text

log = logging.getLogger(__name__)

ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
THUMB_WIDTHS = (64, 128, 192, 256, 320, 384, 512, 768, 1024)
DEFAULT_MAX_UPLOAD = 64 * 1024 * 1024
MAX_UPLOAD_PIXELS = 2 * 89_478_485
"""Pillow's own decompression-bomb limit, enforced here even when something
else lifted ``Image.MAX_IMAGE_PIXELS`` for the process (X/Y/Z plot does)"""
MAX_UPLOAD_FRAMES = 2_000
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".avif", ".jxl", ".tif", ".tiff")
VIDEO_EXTENSIONS = (".mp4", ".webm", ".mov", ".mkv", ".avi", ".gif")
TEXT_EXTENSIONS = (".txt", ".csv", ".json", ".yaml", ".yml", ".cube")  # .cube: 3D LUTs (Post Processing Suite)


class FileError(ValueError):
    """A rejected upload; ``status`` is the HTTP status to answer with."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class FileRecord:
    id: str
    kind: str
    """``upload``, ``output`` or ``network`` (a preview image)"""
    path: str
    name: str
    mime: str
    size: int
    width: int | None
    height: int | None
    created: float
    job_id: str | None = None
    tab: str | None = None
    infotext: str | None = None

    @property
    def is_image(self) -> bool:
        return self.mime.startswith("image/")


@dataclass(frozen=True)
class Sniffed:
    kind: str  # "image" | "video" | "text" | "model"
    mime: str
    ext: str


def sniff(head: bytes, name: str = "") -> Sniffed | None:
    """The content type of a file from its first bytes (``name`` only
    disambiguates plain text). None: not a type the WebUI accepts."""

    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return Sniffed("image", "image/png", ".png")
    if head.startswith(b"\xff\xd8\xff"):
        return Sniffed("image", "image/jpeg", ".jpg")
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return Sniffed("image", "image/gif", ".gif")
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return Sniffed("image", "image/webp", ".webp")
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return Sniffed("video", "video/x-msvideo", ".avi")
    if head.startswith(b"BM"):
        return Sniffed("image", "image/bmp", ".bmp")
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return Sniffed("image", "image/tiff", ".tif")
    if head.startswith((b"\xff\x0a", b"\x00\x00\x00\x0cJXL \r\n\x87\n")):
        return Sniffed("image", "image/jxl", ".jxl")
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in (b"avif", b"avis"):
            return Sniffed("image", "image/avif", ".avif")
        if brand in (b"heic", b"heix", b"mif1", b"msf1"):
            return Sniffed("image", "image/heic", ".heic")
        if brand == b"qt  ":
            return Sniffed("video", "video/quicktime", ".mov")
        return Sniffed("video", "video/mp4", ".mp4")
    if head.startswith(b"\x1aE\xdf\xa3"):
        return Sniffed("video", "video/webm", ".webm")
    if len(head) >= 9 and head[8:9] == b"{" and int.from_bytes(head[:8], "little") < 100 * 1024 * 1024:
        return Sniffed("model", "application/octet-stream", ".safetensors")
    ext = os.path.splitext(name)[1].lower()
    if ext in TEXT_EXTENSIONS and b"\x00" not in head:
        try:
            head.decode("utf-8")
        except UnicodeDecodeError as error:
            if error.start < len(head) - 4:  # not just a character cut at the end
                return None
        return Sniffed("text", "application/json" if ext == ".json" else "text/plain", ext)
    return None


def clean_name(name: str | None) -> str:
    """A display file name: the last path component, without control
    characters, at most 200 characters."""

    base = re.split(r"[\\/]", name or "")[-1]
    base = "".join(char for char in base if char.isprintable()).strip()
    return base[-200:] or "upload"


def encode_cursor(*parts: Any) -> str:
    return base64.urlsafe_b64encode(":".join(str(part) for part in parts).encode()).decode().rstrip("=")


def decode_cursor(cursor: str | None, count: int) -> list[str] | None:
    if not cursor:
        return None
    try:
        parts = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode().split(":")
    except (ValueError, UnicodeDecodeError):
        return None
    if len(parts) != count or any(not _in_sqlite_range(part) for part in parts):
        return None
    return parts


def decode_offset(cursor: str | None) -> int:
    """The offset of an ``encode_cursor(offset)`` cursor; 0 for none or a
    malformed one (``"²".isdigit()`` is true, ``int("²")`` raises)."""

    position = decode_cursor(cursor, 1)
    text = position[0] if position else ""
    return int(text) if text.isascii() and text.isdigit() else 0


def _in_sqlite_range(part: str) -> bool:
    """False for an integer SQLite cannot bind (a crafted cursor: OverflowError)."""

    try:
        return -(2**63) <= int(part) < 2**63
    except ValueError:
        # not an integer (a name key), unless it is one too long for int() (over 4300 digits)
        digits = part.strip().lstrip("+-")
        return not (digits.isascii() and digits.isdigit())


def _like_terms(query: str | None) -> list[str]:
    return [term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") for term in (query or "").split() if term][:8]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    name TEXT NOT NULL,
    mime TEXT NOT NULL,
    size INTEGER NOT NULL,
    width INTEGER,
    height INTEGER,
    created REAL NOT NULL,
    sha256 TEXT,
    job_id TEXT,
    tab TEXT,
    infotext TEXT
);
DROP INDEX IF EXISTS files_kind_created;
CREATE INDEX IF NOT EXISTS files_kind_created_id ON files(kind, created, id);
DROP INDEX IF EXISTS files_kind_path;
CREATE UNIQUE INDEX IF NOT EXISTS files_kind_path_unique ON files(kind, path) WHERE kind <> 'upload';
CREATE INDEX IF NOT EXISTS files_kind_path_lookup ON files(kind, path);
CREATE INDEX IF NOT EXISTS files_sha ON files(sha256);
CREATE INDEX IF NOT EXISTS files_job ON files(job_id, kind, created);
CREATE TABLE IF NOT EXISTS removed (path TEXT PRIMARY KEY);
CREATE INDEX IF NOT EXISTS files_path_nocase ON files(path COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS removed_path_nocase ON removed(path COLLATE NOCASE);
"""
# ``removed``: outputs taken out of the gallery but kept on disk (the output
# folder scan skips them). The NOCASE indexes find an output stored under
# another spelling of its path (see `FileStore._output_at`). One record per
# file, except uploads: the same content uploaded under another name is a
# record of its own (its name) sharing the stored file.

_COLUMNS = "id, kind, path, name, mime, size, width, height, created, job_id, tab, infotext"


def connect(path: Path) -> sqlite3.Connection:
    """A WAL-mode SQLite connection shared between threads (callers lock)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    return db


def close_db(db: sqlite3.Connection) -> None:
    """Checkpoint the WAL cleanly (best-effort) and close ``db``."""

    try:
        if db.in_transaction:
            with contextlib.suppress(sqlite3.Error):
                db.execute("ROLLBACK")
        with contextlib.suppress(sqlite3.Error):
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.ProgrammingError:
        return
    finally:
        with contextlib.suppress(sqlite3.Error):
            db.close()


def readonly_uri(path: str | os.PathLike) -> str:
    """The ``file:`` URI that opens the SQLite database at ``path`` read-only
    (``sqlite3.connect(uri, uri=True)``): ``%``, ``?`` and ``#`` in the path
    are escaped, and a Windows drive path becomes ``file:///C:/…``."""

    return Path(path).resolve().as_uri() + "?mode=ro"


@contextlib.contextmanager
def transaction(db: Any, begin: str = "BEGIN IMMEDIATE") -> Iterator[None]:
    """``begin`` … ``COMMIT`` around the block (the caller holds the
    connection's lock). An error in the block rolls back and is raised as it
    was, even when the rollback fails too; a ``COMMIT`` that fails (a full
    disk, a busy reader) leaves an open transaction, which is rolled back so
    the connection takes the next one."""

    db.execute(begin)
    try:
        yield
    except BaseException:
        if db.in_transaction:
            try:
                db.execute("ROLLBACK")
            except sqlite3.Error:
                log.warning("rolling back a failed transaction failed too", exc_info=True)
        raise
    try:
        db.execute("COMMIT")
    except BaseException:
        if db.in_transaction:
            with contextlib.suppress(sqlite3.Error):
                db.execute("ROLLBACK")
        raise


class FileStore:
    """The id → file index plus the upload, output and thumbnail folders
    under ``root``. ``max_upload_bytes`` may be a callable (a live setting).
    ``outputs/`` holds images the save path did not write (saving off); they
    are temporary, see :meth:`prune_unsaved`."""

    def __init__(self, root: str | os.PathLike, *, max_upload_bytes: int | Callable[[], int] = DEFAULT_MAX_UPLOAD) -> None:
        self.root = Path(os.path.realpath(root))  # outputs are indexed by real path (outputs_dir is compared with them)
        self.uploads_dir = self.root / "uploads"
        self.outputs_dir = self.root / "outputs"
        self.thumbs_dir = self.root / "thumbs"
        self._max_upload = max_upload_bytes
        self._lock = threading.RLock()
        self._db = connect(self.root / "files.db")
        self._ensure_schema()
        self._finalizer = weakref.finalize(self, close_db, self._db)

    def _ensure_schema(self) -> None:
        with self._lock:
            rows = self._db.execute("SELECT type, name FROM sqlite_master WHERE type IN ('table', 'index')").fetchall()
            tables = {name for kind, name in rows if kind == "table"}
            indexes = {name for kind, name in rows if kind == "index"}
            required_tables = {"files", "removed"}
            required_indexes = {
                "files_kind_created_id",
                "files_kind_path_unique",
                "files_kind_path_lookup",
                "files_sha",
                "files_job",
                "files_path_nocase",
                "removed_path_nocase",
            }
            obsolete_indexes = {"files_kind_created", "files_kind_path"}
            if required_tables <= tables and required_indexes <= indexes and not (obsolete_indexes & indexes):
                return
            self._db.executescript(_SCHEMA)

    @property
    def max_upload_bytes(self) -> int:
        limit = self._max_upload
        return int(limit() if callable(limit) else limit)

    @max_upload_bytes.setter
    def max_upload_bytes(self, value: int | Callable[[], int]) -> None:
        self._max_upload = value

    def close(self) -> None:
        with self._lock:
            self._finalizer.detach()
            close_db(self._db)

    # region Records

    def _record(self, row: tuple | None) -> FileRecord | None:
        return FileRecord(*row) if row else None

    def _valid_record(self, record: FileRecord | None) -> FileRecord | None:
        if record is None:
            return None
        if os.path.islink(record.path) or os.path.realpath(record.path) != record.path or not os.path.isfile(record.path):
            return None
        if record.kind == "upload" and path_inside(record.path, str(self.uploads_dir)) is None:
            return None
        return record

    def get(self, file_id: str) -> FileRecord | None:
        """The record of ``file_id`` when its file still exists."""

        if not isinstance(file_id, str) or not ID_RE.match(file_id):
            return None
        with self._lock:
            record = self._record(self._db.execute(f"SELECT {_COLUMNS} FROM files WHERE id = ?", (file_id,)).fetchone())
        return self._valid_record(record)

    def get_many(self, file_ids: Iterable[str]) -> dict[str, FileRecord]:
        """The records of those ``file_ids`` whose files still exist (as
        :meth:`get`, one query per 500 ids)."""

        wanted = list(dict.fromkeys(file_id for file_id in file_ids if isinstance(file_id, str) and ID_RE.match(file_id)))
        found: dict[str, FileRecord] = {}
        for start in range(0, len(wanted), 500):
            chunk = wanted[start : start + 500]
            with self._lock:
                rows = self._db.execute(f"SELECT {_COLUMNS} FROM files WHERE id IN ({','.join('?' * len(chunk))})", chunk).fetchall()
            for row in rows:
                record = self._valid_record(FileRecord(*row))
                if record is not None:
                    found[record.id] = record
        return found

    def _insert(self, record: FileRecord, sha256: str | None = None) -> FileRecord:
        with self._lock:
            self._db.execute(
                f"INSERT INTO files ({_COLUMNS}, sha256) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (record.id, record.kind, record.path, record.name, record.mime, record.size, record.width, record.height, record.created, record.job_id, record.tab, record.infotext, sha256),
            )
        return record

    def _by_path(self, kind: str, path: str) -> FileRecord | None:
        with self._lock:
            return self._record(self._db.execute(f"SELECT {_COLUMNS} FROM files WHERE kind = ? AND path = ?", (kind, path)).fetchone())

    def _output_at(self, path: str) -> FileRecord | None:
        """The output record of the file at ``path``, also when it is stored
        under another spelling of that file: on a volume that ignores case,
        macOS keeps the spelling it is given (two settings can name one
        folder differently). Only ASCII letters are compared without case."""

        record = self._by_path("output", path)
        if record is not None:
            return record
        with self._lock:
            rows = self._db.execute(f"SELECT {_COLUMNS} FROM files WHERE path = ? COLLATE NOCASE AND kind = 'output'", (path,)).fetchall()
        return next((record for record in map(self._record, rows) if record is not None and _same_file(record.path, path)), None)

    def _removed_as(self, path: str) -> list[str]:
        """The ``removed`` rows naming the file at ``path``, under its spelling or another."""

        with self._lock:
            rows = [row[0] for row in self._db.execute("SELECT path FROM removed WHERE path = ? COLLATE NOCASE", (path,)).fetchall()]
        return [row for row in rows if row == path or _same_file(row, path)]

    def delete(self, file_id: str, *, delete_file: bool = False, created_before: float | None = None) -> bool:
        """Forget a file; uploads and unsaved outputs are deleted, other
        outputs only with ``delete_file`` (else the folder scan skips them
        from now on). ``created_before``: only if still that old (a prune;
        a re-upload refreshes the time)."""

        with self._transaction():
            row = self._db.execute("SELECT path, kind, created FROM files WHERE id = ?", (file_id,)).fetchone()
            if row is None or (created_before is not None and row[2] >= created_before):
                return False
            self._db.execute("DELETE FROM files WHERE id = ?", (file_id,))
            for sql, params in (
                ("DELETE FROM gallery_meta WHERE file_id = ?", (file_id,)),
                ("DELETE FROM board_items WHERE file_id = ?", (file_id,)),
                ("DELETE FROM asset_lineage WHERE child_id = ? OR parent_id = ?", (file_id, file_id)),
            ):
                with contextlib.suppress(sqlite3.OperationalError):
                    self._db.execute(sql, params)
            path, kind, _ = row
            remove = delete_file or kind == "upload" or self._unsaved(path, self._outputs_real())
            if kind == "output" and not remove:
                self._db.execute("INSERT OR IGNORE INTO removed (path) VALUES (?)", (path,))
            if kind == "upload":
                # Under the lock: an upload of the same content writes this very path. Kept
                # while that content is still uploaded under another name
                if self._db.execute("SELECT 1 FROM files WHERE kind = 'upload' AND path = ? LIMIT 1", (path,)).fetchone() is None:
                    Path(path).unlink(missing_ok=True)
                remove = False
        if remove:
            Path(path).unlink(missing_ok=True)
        self.delete_thumbnails(file_id)
        return True

    def delete_thumbnails(self, file_id: str) -> None:
        """Remove the cached thumbnails of ``file_id`` (by name: a glob would
        list the whole thumbnail folder for each file)."""

        for width in THUMB_WIDTHS:
            (self.thumbs_dir / f"{file_id}-{width}.webp").unlink(missing_ok=True)

    # endregion

    # region Uploads

    def add_upload(self, chunks: Iterable[bytes], name: str | None = None) -> FileRecord:
        """Store an upload (its bytes, in chunks). Raises :class:`FileError`
        (413 when too large, 415 for a type the WebUI does not accept)."""

        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        head = b""
        temp = self.uploads_dir / f".incoming-{secrets.token_hex(8)}"
        try:
            with open(temp, "wb") as file:
                for chunk in chunks:
                    size += len(chunk)
                    if size > self.max_upload_bytes:
                        raise FileError(f"the file is larger than {self.max_upload_bytes // (1024 * 1024)} MB", 413)
                    if len(head) < 64:
                        head += chunk[: 64 - len(head)]
                    digest.update(chunk)
                    file.write(chunk)
            if size == 0:
                raise FileError("the file is empty")
            name = clean_name(name)
            sniffed = sniff(head, name)
            if sniffed is None:
                raise FileError("unsupported file type", 415)
            width = height = None
            if sniffed.kind == "image":
                width, height = _image_size(temp, max_pixels=MAX_UPLOAD_PIXELS)
            sha = digest.hexdigest()
            target = self.uploads_dir / sha[:2] / f"{sha}{sniffed.ext}"
            moved_to_target = False
            try:
                with self._transaction():  # one record per content and name, even for concurrent uploads
                    existing = self._upload_by_sha(sha, name)
                    if existing is not None:
                        return self._touch(existing)  # a re-upload: not old enough to prune
                    stored = self._upload_by_sha(sha)
                    if stored is not None:  # the same content under another name: its file, this name
                        target = Path(stored.path)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(temp, target)
                        moved_to_target = True
                        self._db.execute("DELETE FROM files WHERE kind = 'upload' AND path = ?", (str(target),))  # its file was gone
                    record = FileRecord(secrets.token_urlsafe(16), "upload", str(target), name, sniffed.mime, size, width, height, time.time())
                    return self._insert(record, sha)
            except BaseException:
                if moved_to_target:
                    target.unlink(missing_ok=True)
                raise
        finally:
            temp.unlink(missing_ok=True)

    def add_upload_bytes(self, data: bytes, name: str | None = None) -> FileRecord:
        return self.add_upload([data], name)

    def _touch(self, record: FileRecord) -> FileRecord:
        now = time.time()
        with self._lock:
            self._db.execute("UPDATE files SET created = ? WHERE id = ?", (now, record.id))
        return dataclasses.replace(record, created=now)

    def _upload_by_sha(self, sha: str, name: str | None = None) -> FileRecord | None:
        """An upload of this content (``name``: under that name) whose file still exists."""

        query = f"SELECT {_COLUMNS} FROM files WHERE kind = 'upload' AND sha256 = ?" + (" AND name = ?" if name is not None else "")
        with self._lock:
            record = self._record(self._db.execute(query, (sha, name) if name is not None else (sha,)).fetchone())
        return self._valid_record(record)

    def resolve_upload(self, node: Any, file_id: str) -> Any:
        """``remi.ui`` upload resolver: a PIL image for image inputs (with
        the original name as ``image.filename``), the path for ``File``."""

        from remi.ui import ParamError, Sentence
        from remi.ui.nodes import File

        record = self.get(file_id)
        if record is None or record.kind not in ("upload", "output"):
            raise ParamError(Sentence("unknown upload {{id}}", id=repr(file_id)))
        if isinstance(node, File):
            return record.path
        if not record.is_image:
            raise ParamError("the upload is not an image")
        from modules import images

        image = images.read(record.path)
        try:
            image.load()
        except BaseException:
            image.close()  # a truncated file: do not leave it open
            raise
        image.filename = record.name
        return image

    def check_upload(self, node: Any, file_id: str) -> str:
        """Upload resolver for validation only: the id, if it names a file."""

        from remi.ui import ParamError, Sentence
        from remi.ui.nodes import File

        record = self.get(file_id)
        if record is None or record.kind not in ("upload", "output"):
            raise ParamError(Sentence("unknown upload {{id}}", id=repr(file_id)))
        if not isinstance(node, File) and not record.is_image:
            raise ParamError("the upload is not an image")
        return file_id

    def prune_uploads(self, max_age: float, keep: set[str] | frozenset[str] = frozenset()) -> int:
        """Delete uploads older than ``max_age`` seconds, except ``keep``, and
        partial uploads (``.incoming-*``) a killed process left that long ago."""

        cutoff = time.time() - max_age
        with self._lock:
            rows = self._db.execute("SELECT id FROM files WHERE kind = 'upload' AND created < ?", (cutoff,)).fetchall()
        removed = 0
        for (file_id,) in rows:
            if file_id not in keep and self.delete(file_id, created_before=cutoff):
                removed += 1
        for partial in self.uploads_dir.glob(".incoming-*"):
            with contextlib.suppress(OSError):  # finished or removed meanwhile
                if partial.stat().st_mtime < cutoff:
                    partial.unlink()
        return removed

    def prune_unsaved(self, max_age: float, keep: set[str] | frozenset[str] = frozenset()) -> int:
        """Delete outputs older than ``max_age`` seconds that the save path
        did not write (``outputs/``: generated with saving turned off), except
        ``keep`` (ids a queued job still reads), and stale thumbnail temps."""

        cutoff = time.time() - max_age
        with self._lock:
            rows = self._db.execute("SELECT id, path FROM files WHERE kind = 'output' AND created < ?", (cutoff,)).fetchall()
        outputs = self._outputs_real()
        removed = sum(1 for file_id, path in rows if file_id not in keep and self._unsaved(path, outputs) and self.delete(file_id))
        thumb_cutoff = time.time() - min(max_age, 3600.0)
        if self.thumbs_dir.is_dir():
            for partial in self.thumbs_dir.glob("*.tmp"):
                with contextlib.suppress(OSError):
                    if partial.stat().st_mtime < thumb_cutoff:
                        partial.unlink()
        return removed

    def _outputs_real(self) -> str:
        """The real path of ``outputs/`` (a link to another disk, say): outputs are stored under theirs."""

        return os.path.realpath(self.outputs_dir)

    def _unsaved(self, path: str, outputs: str) -> bool:
        """The output at ``path`` is one the save path did not write (in ``outputs/``)."""

        parent = Path(path).parent
        return parent == self.outputs_dir or str(parent) == outputs

    # endregion

    # region Outputs

    def register_output(self, path: str | os.PathLike, *, job_id: str | None = None, tab: str | None = None, infotext: str | None = None, width: int | None = None, height: int | None = None, created: float | None = None) -> FileRecord:
        """Index a generated file (idempotent per path). Its infotext is kept up
        to :data:`~remi.image_meta.MAX_INFOTEXT_CHARS` (an image's own text can be any size),
        with its parameters line."""

        if infotext is not None:
            infotext = utf8_text(clip_infotext(infotext))
        raw_path = os.fspath(path)
        if os.path.islink(raw_path):
            raise FileError("symbolic links cannot be registered as outputs", 400)
        real = os.path.realpath(raw_path)
        from remi.setting_values import output_dirs

        real_bases = [os.path.realpath(os.fspath(base)) for base in (self.root, self.outputs_dir, *output_dirs()) if base]
        base_ids: set[tuple[int, int]] = set()
        for base in real_bases:
            try:
                st = os.stat(base)
            except OSError:
                continue
            base_ids.add((st.st_dev, st.st_ino))

        def _inside_real_bases(candidate_real: str) -> bool:
            if any(path_inside(candidate_real, base) is not None for base in real_bases):
                return True
            if not base_ids:
                return False
            cand_path = Path(candidate_real)
            for ancestor in (cand_path, *cand_path.parents):
                try:
                    st = os.stat(ancestor)
                except OSError:
                    continue
                if (st.st_dev, st.st_ino) in base_ids:
                    return True
            return False

        if not _inside_real_bases(real):
            p_obj = Path(raw_path)
            if p_obj.is_absolute():
                cur = os.path.realpath(p_obj.anchor)
                parts = p_obj.parts[1:]
            else:
                cur = os.path.realpath(os.getcwd())
                parts = p_obj.parts
            for part in parts:
                if part == ".":
                    continue
                if part == "..":
                    cur = os.path.realpath(os.path.dirname(cur))
                    continue
                nxt = os.path.join(cur, part)
                if os.path.islink(nxt) and _inside_real_bases(cur):
                    raise FileError("symbolic link escapes output directory", 400)
                cur = os.path.realpath(nxt)
        path = real
        stale = self._removed_as(path)
        # A first look, so that the file is read (stat, type, size) before the transaction
        existing = self._output_at(path)
        record = None if existing is not None and not _overwritten(existing, job_id) else self._new_output(path, job_id, tab, infotext, width, height, created)
        # The lookup and the insert in one write transaction: a job saving one spelling of a
        # file while the folder scan (or the worker process, with a store of its own) indexes
        # another each found nothing and inserted a row of its own
        with self._transaction():
            self._db.executemany("DELETE FROM removed WHERE path = ?", [(removed,) for removed in stale])
            existing = self._output_at(path)
            if existing is not None and _overwritten(existing, job_id):
                # Another job's output: this job wrote over it (a fixed file name, "Override"). A record
                # of its own, so the old job's file id neither shows nor deletes this image
                self._forget_row(existing)
                existing = None
            if existing is None:
                record = record or self._new_output(path, job_id, tab, infotext, width, height, created)
                try:
                    return self._insert(record)
                except sqlite3.IntegrityError:  # its path or its id is taken: the row that has the path, or a new id
                    existing = self._output_at(path)
                    if existing is None:
                        return self._insert(dataclasses.replace(record, id=secrets.token_urlsafe(16)))
            return self._claim(existing, job_id, tab, infotext, width, height, created) if job_id and existing.job_id is None else existing

    @contextlib.contextmanager
    def _transaction(self) -> Iterator[None]:
        """A write transaction under the lock: no other connection (the worker
        process's store, the gallery) writes between its reads and its writes."""

        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                if self._db.in_transaction:
                    self._db.execute("ROLLBACK")
                raise
            self._db.execute("COMMIT")

    def _forget_row(self, record: FileRecord) -> None:
        """Drop an output's record (and its gallery row and thumbnails), keeping the file."""

        with self._lock:
            self._db.execute("DELETE FROM files WHERE id = ?", (record.id,))
            try:  # the gallery's side table (remi.library), when it exists
                self._db.execute("DELETE FROM gallery_meta WHERE file_id = ?", (record.id,))
            except sqlite3.OperationalError:
                pass
        self.delete_thumbnails(record.id)

    def _new_output(self, path: str, job_id: str | None, tab: str | None, infotext: str | None, width: int | None, height: int | None, created: float | None) -> FileRecord:
        stat = os.stat(path)
        sniffed = sniff(_read_head(path), path)
        mime = sniffed.mime if sniffed else "application/octet-stream"
        if sniffed and sniffed.kind == "image" and (width is None or height is None):
            try:
                width, height = _image_size(path)
            except FileError:
                pass
        return FileRecord(secrets.token_urlsafe(16), "output", path, os.path.basename(path), mime, stat.st_size, width, height, created or stat.st_mtime, job_id, tab, infotext)

    def _claim(self, record: FileRecord, job_id: str, tab: str | None, infotext: str | None, width: int | None, height: int | None, created: float | None = None) -> FileRecord:
        """Give an output the folder scan indexed while its job still ran
        (no job, no tab) the job's fields, and let the gallery file it again
        (its category follows the tab). The job's own infotext and size win:
        the scan may have indexed an older file the job overwrote (so its
        time is the job's file's too, or the gallery files it weeks back)."""

        try:
            stat = os.stat(record.path)
            size, mtime = stat.st_size, stat.st_mtime
        except OSError:
            size, mtime = record.size, record.created
        fields = {"job_id": job_id, "tab": tab or record.tab, "infotext": infotext or record.infotext, "size": size, "width": width or record.width, "height": height or record.height, "created": created or mtime}
        with self._lock:
            self._db.execute("UPDATE files SET job_id = ?, tab = ?, infotext = ?, size = ?, width = ?, height = ?, created = ? WHERE id = ?", (*fields.values(), record.id))
            try:  # the gallery's side table (remi.library), when it exists; a starred row keeps its category
                self._db.execute("DELETE FROM gallery_meta WHERE file_id = ? AND starred = 0", (record.id,))
            except sqlite3.OperationalError:
                pass
        if size != record.size:
            self.delete_thumbnails(record.id)  # of the overwritten file
        return dataclasses.replace(record, **fields)

    def save_output_image(self, image: Image.Image, infotext: str | None, **fields: Any) -> FileRecord:
        """Store an image that the normal save path did not write (sample
        saving off) as PNG with its infotext, and index it."""

        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        metadata = PngImagePlugin.PngInfo()
        texts = {key: value for key, value in (image.info or {}).items() if isinstance(key, str) and isinstance(value, str)}
        if infotext:
            texts.setdefault("parameters", infotext)
        for key, value in texts.items():
            metadata.add_text(key, value)
        path = self.outputs_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(4)}.png"
        image.save(path, pnginfo=metadata if texts else None)
        return self.register_output(path, infotext=infotext, width=image.width, height=image.height, **fields)

    def list_outputs(self, *, cursor: str | None = None, limit: int = 50, query: str | None = None, tab: str | None = None) -> tuple[list[FileRecord], str | None, int]:
        """Newest first: ``(records, next_cursor, total)``; ``query`` terms
        must all appear in the infotext or the file name."""

        where, args = ["kind = 'output'"], []
        if tab:
            where.append("tab = ?")
            args.append(tab)
        for term in _like_terms(query):
            where.append("(infotext LIKE ? ESCAPE '\\' OR name LIKE ? ESCAPE '\\')")
            args += [f"%{term}%", f"%{term}%"]
        filters = " AND ".join(where)
        page_where, page_args = filters, list(args)
        position = decode_cursor(cursor, 2)
        if position is not None:
            try:
                created, rowid = float(position[0]), int(position[1])
            except ValueError:
                position = None
            else:
                if math.isfinite(created) and -(2**63) <= rowid < 2**63:
                    page_where += " AND (created < ? OR (created = ? AND rowid < ?))"
                    page_args += [created, created, rowid]
        limit = min(max(int(limit), 1), 200)
        with self._lock:
            total = self._db.execute(f"SELECT COUNT(*) FROM files WHERE {filters}", args).fetchone()[0]
            rows = self._db.execute(f"SELECT {_COLUMNS}, rowid FROM files WHERE {page_where} ORDER BY created DESC, rowid DESC LIMIT ?", [*page_args, limit + 1]).fetchall()
        records = [FileRecord(*row[:-1]) for row in rows[:limit]]
        next_cursor = encode_cursor(rows[limit - 1][8], rows[limit - 1][-1]) if len(rows) > limit else None
        return records, next_cursor, total

    def scan_outputs(self, directories: Iterable[str | os.PathLike], *, limit: int = 50_000) -> int:
        """Index images found in ``directories`` that are not indexed yet
        (outputs from before Remi, or saved by other tools); returns how many."""

        from modules import images

        added = 0
        for path, dir_real in _walk_images_under(directories):
            if added >= limit:
                break
            if os.path.islink(path):
                log.debug("not indexing symlink %r", path)
                continue
            real = os.path.realpath(path)
            if path_inside(real, dir_real) is None:
                log.debug("not indexing %r: resolves outside %r", real, dir_real)
                continue
            if utf8_text(real) != real:
                # a file name that is not UTF-8 (Linux keeps its bytes as surrogates): no row can hold it
                log.debug("not indexing %r: its name is not UTF-8", real)
                continue
            existing = self._output_at(real)
            if existing is not None:
                with self._lock, contextlib.suppress(sqlite3.OperationalError):
                    if self._db.execute("UPDATE gallery_meta SET missing_since = NULL WHERE file_id = ? AND missing_since IS NOT NULL", (existing.id,)).rowcount:
                        added += 1
                continue
            if self._was_removed(real):
                continue
            infotext = width = height = None
            try:
                with Image.open(real) as image:
                    width, height = image.size
                    infotext, _ = images.read_info_from_image(image)
            except Exception:
                continue
            try:
                self.register_output(real, infotext=infotext, width=width, height=height)
            except (OSError, sqlite3.Error):  # gone or unreadable meanwhile: the images after it are still indexed
                log.debug("not indexing %r", real, exc_info=True)
                continue
            added += 1
        return added

    def _was_removed(self, path: str) -> bool:
        return bool(self._removed_as(path))

    def is_removed(self, path: str | os.PathLike) -> bool:
        """``path`` was taken out of the gallery (kept on disk)."""

        return self._was_removed(os.path.realpath(path))

    # endregion

    # region Network previews

    def register_path(self, path: str | os.PathLike, kind: str = "network") -> FileRecord | None:
        """Index an existing file under an id stable while the file is
        unchanged (path, size and mtime: a changed file gets a new id, so
        cached copies are not reused); None when it is gone."""

        path = os.path.realpath(path)
        if utf8_text(path) != path:  # a file name that is not UTF-8 (Linux): no row can hold it
            return None
        try:
            stat = os.stat(path)
        except OSError:
            return None
        existing = self._by_path(kind, path)
        if existing is not None:
            if existing.size == stat.st_size and existing.created == stat.st_mtime:
                return existing
            self.delete(existing.id)
        sniffed = sniff(_read_head(path), path)
        record = FileRecord(_stable_id(kind, path, stat.st_size, stat.st_mtime_ns), kind, path, os.path.basename(path), sniffed.mime if sniffed else "application/octet-stream", stat.st_size, None, None, stat.st_mtime)
        try:
            return self._insert(record)
        except sqlite3.IntegrityError:
            return self._by_path(kind, path)

    # endregion

    # region Thumbnails

    def thumbnail(self, record: FileRecord, width: int | None) -> Path:
        """A cached WebP thumbnail ``width`` wide (snapped to a fixed set)."""

        if not record.is_image:
            raise FileError("not an image", 415)
        width = snap_width(width)
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)
        target = self.thumbs_dir / f"{record.id}-{width}.webp"
        source_mtime = os.path.getmtime(record.path)
        if target.is_file() and target.stat().st_mtime >= source_mtime:
            return target
        temp = target.with_suffix(f".{secrets.token_hex(4)}.tmp")
        try:
            with Image.open(record.path) as image:
                if getattr(image, "is_animated", False):
                    image.seek(0)
                image.draft("RGB", (width, width * 4))
                image = image.convert("RGBA" if image.mode in ("RGBA", "LA", "P") else "RGB")
                if image.width > width:
                    image.thumbnail((width, max(1, round(image.height * width / image.width))), Image.Resampling.LANCZOS)
                image.save(temp, format="WEBP", quality=82, method=4)
            os.replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)
        return target

    # endregion


def snap_width(width: int | None) -> int:
    if not width or width <= 0:
        return 256
    return next((size for size in THUMB_WIDTHS if size >= width), THUMB_WIDTHS[-1])


def _stable_id(kind: str, path: str, size: int, mtime_ns: int) -> str:
    return "n" + base64.urlsafe_b64encode(hashlib.sha256(f"{kind}\0{path}\0{size}\0{mtime_ns}".encode()).digest()[:18]).decode()


def _read_head(path: str, size: int = 64) -> bytes:
    try:
        with open(path, "rb") as file:
            return file.read(size)
    except OSError:
        return b""


def _image_size(path: str | os.PathLike, *, max_pixels: int | None = None) -> tuple[int, int]:
    """Width and height from the header (nothing is decoded). ``max_pixels``:
    refuse larger images (and animations of more frames than
    :data:`MAX_UPLOAD_FRAMES`) whatever ``Image.MAX_IMAGE_PIXELS`` is."""

    try:
        with Image.open(path) as image:  # raises DecompressionBombError past Pillow's pixel limit
            width, height = image.size
            if max_pixels is not None and (width * height > max_pixels or getattr(image, "n_frames", 1) > MAX_UPLOAD_FRAMES):
                raise Image.DecompressionBombError(f"{width}x{height}")
            return width, height
    except Image.DecompressionBombError as error:
        raise FileError("the image is too large", 413) from error
    except Exception as error:
        # Pillow's message names the file (a path in the data folder): logged,
        # never sent to the client
        log.debug("not a readable image: %s", error)
        raise FileError("not a readable image", 415) from error


def _overwritten(record: FileRecord, job_id: str | None) -> bool:
    """``record`` is another job's output that the job ``job_id`` wrote over."""

    return bool(job_id) and record.job_id not in (None, job_id)


def _same_file(a: str, b: str) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def path_inside(path: str, folder: str) -> str | None:
    """The part of ``path`` below ``folder`` (``""``: the folder itself), also
    when ``path`` spells the folder another way on a volume that ignores
    case (macOS and Windows keep the spelling they are given); None when it
    is not inside. Both are compared as given: resolve them first if links
    matter."""

    prefix = folder.rstrip(os.sep)
    if path in (folder, prefix):
        return ""
    if path.startswith(prefix + os.sep):
        return path[len(prefix) + 1 :]
    lowered = path.lower()
    if len(lowered) != len(path) or not (lowered == prefix.lower() or lowered.startswith(prefix.lower() + os.sep)):
        return None
    if not _same_file(path[: len(prefix)], prefix):
        return None
    return path[len(prefix) + 1 :]


def _walk_images_under(directories: Iterable[str | os.PathLike]) -> Iterator[tuple[str, str]]:
    """The ``(image_path, real_directory)`` pairs below ``directories``, each
    folder walked once and symlinks skipped."""

    seen: set[tuple[int, int]] = set()

    def first_visit(path: str) -> bool:
        try:
            stat = os.stat(path, follow_symlinks=False)  # a link is not walked: its own identity
        except OSError:
            return False
        key = (stat.st_dev, stat.st_ino)
        if key in seen:
            return False
        seen.add(key)
        return True

    for directory in directories:
        if not directory or not os.path.isdir(directory):
            continue
        real = os.path.realpath(directory)
        if not first_visit(real):
            continue
        for current, dirs, files in os.walk(real):
            dirs[:] = [name for name in dirs if first_visit(os.path.join(current, name))]
            for name in files:
                if name.lower().endswith(IMAGE_EXTENSIONS):
                    candidate = os.path.join(current, name)
                    if not os.path.islink(candidate):
                        yield candidate, real


def _walk_images(directories: Iterable[str | os.PathLike]) -> Iterator[str]:
    """The image files below ``directories``, each folder walked once: two
    settings can name one folder in different case (macOS keeps the case it
    is given, realpath too), or one folder inside another."""

    for path, _root in _walk_images_under(directories):
        yield path
