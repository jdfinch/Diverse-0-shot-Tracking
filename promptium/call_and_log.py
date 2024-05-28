
import functools
import inspect


def call_and_log(f, filepath=None, settings=None, **settings_kwargs):
    """
    Wrap a function f so that each call will additionally log the input and output to f.

    :param f: function to wrap and call
    :param filepath: file to append to with log info
    :param settings: object whose __dict__ represents additional settings (or callable producing settings dict)
    :param settings_kwargs: additional settings to log
    :return: function wrapping f (called the same as f would be called)
    """
    signature = inspect.signature(f)
    @functools.wraps(f)
    def fn_with_logging(*args, **kwargs):
        result = f(*args, **kwargs)
        binding_args = signature.bind(*args, **kwargs).arguments
        if callable(settings):
            object_settings = settings()
        else:
            object_settings = settings.__dict__ if settings else {}
        all_args = {**settings_kwargs, **object_settings, **binding_args}
        log_string = '\n'.join((
            f">> {f.__name__}({', '.join((f'{p} = {a}' for p, a in all_args.items()))})",
            f'{result}',
            '',
            ''
        ))
        if filepath:
            with open(filepath, 'a+') as file:
                file.write(log_string)
        else:
            print(log_string)
        return result
    return fn_with_logging


if __name__ == '__main__':

    def foo(a, b, **kwargs):
        return a + b + sum(kwargs.values())

    log_foo = call_and_log(foo)
    log_foo(1, 2)
    log_foo(3, 4)
    log_foo(5, 6, c=3, d=9)