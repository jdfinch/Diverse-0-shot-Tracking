"""
Utilities for maintaining a type-hinted collection of attributes with easy attribute value swapping.
Two utilities are provided:

`settings` decorates a method to automatically fill parameters with self attributes of the same name, but ONLY when arguments are NOT passsed to those parameters.

`replace` is an in-place (mutating) version of dataclasses.replace, and can be used as a context manager to undo the mutations (puts back the attributes entered with) upon exiting the context.
"""

from dataclasses import replace
import functools
import inspect
import contextlib
import sys
import typing as T

F1 = T.TypeVar('F1')
def settings(fn:F1=None, /, **params) -> F1:
    """
    Decorator for automatically filling parameters with self attributes of the same name,
     but ONLY when arguments are NOT passsed to those parameters.

    :param fn: function to decorate
    :param params: optionally, parameters to include (True) or exclude (False)
    :return: wrapped function
    """
    if fn is None:
        return functools.partial(settings, **params)
    else:
        def wrapper(obj, *args, **kwargs):
            null = object()
            mode = 'select' if any(params.values()) else 'filter'
            signature = inspect.signature(fn)
            bound = signature.bind(obj, *args, **kwargs)
            for name, value in signature.parameters.items():
                value = getattr(obj, name, obj.get(name, null) if isinstance(obj, dict) else null)
                if value is not null and name not in bound.arguments and (
                    mode == 'select' and params.get(name) or
                    mode == 'filter' and params.get(name, True)
                ):
                    bound.arguments[name] = value
            result = fn(*bound.args, **bound.kwargs)
            return result
        return wrapper


@contextlib.contextmanager
def temporary_update(obj, originals):
    yield
    obj.__dict__.update(originals)


def replace_inplace(obj, **kwargs):
    originals = {k: obj.__dict__[k] for k in kwargs}
    obj.__dict__.update(kwargs)
    context_manager = temporary_update(obj, originals)
    return context_manager


sys.modules[__name__].__dict__.update(replace=replace_inplace)



if __name__ == '__main__':

    import dataclasses

    @dataclasses.dataclass
    class Foo:
        x: int
        y: str
        z: list[str]

        @settings
        def show(self, y=None):
            return f"x={self.x}, y={y}, z={self.z}"


    foo = Foo(1, '3', ['4'])
    print(f'{foo = }')

    replace(foo, x=2)
    print(f'{foo = }')

    with replace(foo, x=9, z=['9']):
        print(f'    {foo = }')
        with replace(foo, y='9'):
            print(f'        {foo = }')
        print(f'    {foo = }')

    print(f'{foo = }')
    print()

    print(f'{foo.show() = }')
    print(f'{foo.show("hello world") = }')







