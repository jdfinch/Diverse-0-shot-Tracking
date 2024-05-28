
import contextlib
from io import StringIO
import sys


@contextlib.contextmanager
def shush():
    save_stdout, save_stderr = sys.stdout, sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    try:
        yield
    finally:
        sys.stdout = save_stdout
        sys.stderr = save_stderr