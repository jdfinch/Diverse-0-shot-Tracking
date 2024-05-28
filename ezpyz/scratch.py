import dataclasses
import functools
import abc
import json
import csv
import pickle
import io
import typing as T

import ezpyz as ez


class DataMeta(abc.ABCMeta):

    formats = {}
    extensions = None

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        extensions = getattr(cls, 'extensions', None)
        super_extensions = getattr(super(cls, cls), 'extensions', None)
        if extensions is super_extensions:
            cls.formats[f'.{name.lower()}'] = cls
        else:
            cls.formats.update({
                f'.{ext}' if not ext.startswith('.') else ext: cls
                for ext in cls.extensions
            })


@dataclasses.dataclass
class Data(abc.ABC, metaclass=DataMeta):
    file: ez.filelike | None = None

    serialized_in_binary: T.ClassVar[bool] = False

    @classmethod
    @abc.abstractmethod
    def deserialize(cls, string):
        ...

    @abc.abstractmethod
    def serialize(self: ...):
        ...

    def save(self: ..., file, *args, **kwargs):
        file = file.File(file, format=type(self))
        file.save(self, *args, **kwargs)

    @classmethod
    def load(cls, file, *args, **kwargs):
        file = file.File(file, format=cls)
        obj = file.load(*args, **kwargs)
        return obj


class Text(Data):
    extensions = ['txt', 'text', 'log', 'out', 'err']

    @classmethod
    def deserialize(cls, string):
        return string

    def serialize(self: ...):
        return str(self)


class Bytes(Data):
    serialized_in_binary = True

    extensions = ['bytes', 'bin', 'binary', 'b']

    @classmethod
    def deserialize(cls, string):
        return string

    def serialize(self: ...):
        return bytes(self)


class JSON(Data):
    extensions = ['json', 'jsonl']

    @classmethod
    def deserialize(cls, string, *args, **kwargs):
        deserialized = json.loads(string, *args, **kwargs)
        if cls is not JSON:
            deserialized = cls(**deserialized)  # noqa
        return deserialized

    def serialize(self: ..., *args, **kwargs):
        if isinstance(self, (dict, list, tuple, str, int, float, bool, type(None))):
            obj = self
        else:
            obj = vars(self)
        return json.dumps(obj, *args, **kwargs)


class CSV(Data):
    extensions = ['csv', 'tsv']

    @classmethod
    def deserialize(cls, string, *args, **kwargs):
        stream = io.StringIO(string)
        reader = csv.reader(stream, *args, **kwargs)
        return list(reader)

    def serialize(self: ..., *args, **kwargs):
        stream = io.StringIO()
        writer = csv.writer(stream, *args, **kwargs)
        writer.writerows(self)  # noqa
        return stream.getvalue()


class Pickle(Data):
    extensions = ['pkl', 'pickle', 'pckl']
    serialized_in_binary = True

    @classmethod
    def deserialize(cls, string, *args, **kwargs):
        return pickle.loads(string, *args, **kwargs)

    def serialize(self: ..., *args, **kwargs):
        return pickle.dumps(self, *args, **kwargs)






f = Data('foo/bar.txt')
print(f, DataMeta.formats)