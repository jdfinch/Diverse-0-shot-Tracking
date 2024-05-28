

from dataclasses import dataclass


@dataclass
class Foo:
    a: int = None
    b: str = None
    c: list[float] = None

@dataclass
class Bar(Foo):
    d: set[str] = None
    e: dict[str, str] = None

@dataclass
class Bat(Bar):
    g: str = None


ball = Bat(1, '2', [3], {'4'}, {'5': '6'}, '7')
print(ball)