
import atexit
import ezpyz as ez
import dataclasses as dc
import abc
import typing as T

C = T.TypeVar('C')


@dc.dataclass
class Store(ez.Data, abc.ABC):

    @abc.abstractmethod
    def merge(self:C, existing:C):
        pass

    def __post_init__(self):
        super().__post_init__()
        if self.file and self.file.path and self.file.path.exists():
            existing = self.load(self.file)
            self.merge(existing)
        atexit.register(self.save)