
import dataclasses as dc



class Table:
    def __init__(self):
        cls = type(self)
        ...


@dc.dataclass
class Column:
    name: str

