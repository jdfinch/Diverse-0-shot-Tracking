
import traceback


class crashless:

    def __init__(self, crashing_is_allowed=False):
        self.crashing_is_allowed = crashing_is_allowed

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            if self.crashing_is_allowed:
                if issubclass(exc_type, Exception):
                    traceback.print_exception(exc_type, exc_val, exc_tb)
                return True
