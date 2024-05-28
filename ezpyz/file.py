from __future__ import annotations

import ezpyz.format as fmt

import io
import pathlib
import contextlib
import datetime

import typing as T


filelike: T.TypeAlias = T.Union[str, pathlib.Path, io.IOBase, 'File']
class Format(T.Protocol):
    def serialize(obj: ...) -> str|bytes: ...
    @classmethod
    def deserialize(cls, string:str|bytes): ...
formatlike = T.Union[str, type[Format], None]


class File:
    def __init__(self,
        file_or_path: filelike=None,
        format:formatlike=None
    ):
        if isinstance(file_or_path, io.IOBase):
            path = file_or_path.name if hasattr(file_or_path, 'name') else None
            file = file_or_path
        elif isinstance(file_or_path, File):
            path = file_or_path.path
            file = None
            format = file_or_path.format
        else:
            path = file_or_path
            file = None
        if isinstance(format, str):
            if not format.startswith('.'):
                format = '.' + format
            format = fmt.Savable.formats[format]
        self.path: pathlib.Path|None = path and pathlib.Path(path)
        if isinstance(self.path, pathlib.Path):
            if hasattr(file, 'mode'):
                backup_format = fmt.Bytes if 'b' in file.mode else fmt.Text
            else:
                backup_format = fmt.Text
        else:
            backup_format = fmt.Text
        format = self.formats.get(self.path.suffix, backup_format) if format is None else format
        self.format:type[fmt.Savable] = format
        self.file:io.IOBase|None = file

    SEEK_SET = io.SEEK_SET
    SEEK_CUR = io.SEEK_CUR
    SEEK_END = io.SEEK_END

    formats:dict[str, fmt.Savable] = fmt.SavableMeta.formats

    def save(self, obj, *args, **kwargs):
        serialized = self.format.serialize(obj, *args, **kwargs) # noqa
        self.write(serialized)

    def log(self, obj, *args, **kwargs):
        serialized = self.format.serialize(obj, *args, **kwargs) # noqa
        self.append(serialized)

    def load(self, *args, **kwargs):
        serialized = self.read()
        obj = self.format.deserialize(serialized, *args, **kwargs) # noqa
        return obj

    @property
    def closed(self):
        return self.file is None or self.file.closed

    @property
    def binary(self):
        return self.format and self.format.serialized_in_binary

    def open(self, path:filelike=None):
        if path is not None:
            path = pathlike_to_path(path)
            self.path = path
            self.format = self.format or self.formats.get(self.path.suffix, self.format)
        if self.path is None:
            if self.binary:
                self.file = io.BytesIO()
            else:
                self.file = io.StringIO()
        else:
            if not self.path.exists():
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.touch()
            if self.binary:
                self.file = open(self.path, 'rb+')
            else:
                self.file = open(self.path, 'r+')
        return self

    def close(self):
        if self.file is not None:
            self.file.close()
        self.file = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.file is not None:
            self.file.close()
        self.file = None

    @contextlib.contextmanager
    def _open_if_closed(self, error_if_does_not_exist=False):
        was_closed = self.closed
        if was_closed:
            if error_if_does_not_exist and not self.path.exists():
                raise FileNotFoundError(self.path)
            self.open()
        yield
        if was_closed:
            self.close()

    def read(self, n=None):
        with self._open_if_closed(error_if_does_not_exist=True):
            if n is None:
                self.file.seek(0)
                return self.file.read()
            else:
                return self.file.read(n)

    def append(self, content):
        with self._open_if_closed():
            self.file.seek(0, 2)
            self.file.write(content)

    def write(self, content, start=0):
        with self._open_if_closed():
            if not isinstance(start, int) or start < 0:
                self.file.seek(0, 2)
            else:
                self.file.seek(start)
            self.file.write(content)
            self.file.truncate()

    def rewrite(self, content, start=None):
        with self._open_if_closed():
            if start is not None:
                self.file.seek(start)
            self.file.write(content)

    def seek(self, offset, whence=0):
        self.file.seek(offset, whence)

    def tell(self):
        return self.file.tell()

    def truncate(self, size=None):
        with self._open_if_closed():
            self.file.truncate(size)

    def flush(self):
        self.file.flush()

    def seekable(self):
        return self.file and self.file.seekable()

    def writable(self):
        return self.file and self.file.writable()

    def modified_time(self):
        if self.path is not None and self.path.exists():
            return datetime.datetime.fromtimestamp(self.path.stat().st_mtime)
        return None

    def accessed_time(self):
        if self.path is not None and self.path.exists():
            return datetime.datetime.fromtimestamp(self.path.stat().st_atime)
        return None

    def created_time(self):
        if self.path is not None and self.path.exists():
            return datetime.datetime.fromtimestamp(self.path.stat().st_ctime)
        return None

    def __str__(self):
        return f"{type(self).__name__}({self.path})"
    __repr__ = __str__


def pathlike_to_path(path:filelike):
    if isinstance(path, pathlib.Path):
        return path
    elif hasattr(path, 'name'):
        return pathlib.Path(path.name)
    elif isinstance(path, str):
        return pathlib.Path(path)
    elif hasattr(path, 'path'):
        return path.path
    else:
        return pathlib.Path(str(path))

file = File('hello/world.txt', format=fmt.JSON)