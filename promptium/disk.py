import json
import pathlib

def read_file(file):
    """
    Read a file flexibly.

    :param file:  file as object, path string, .read()-able, or callable.
    :return: str content of file (empty string if file not found)
    """
    if isinstance(file, (str, pathlib.Path)):
        if pathlib.Path(file).exists():
            with open(file) as f:
                return f.read()
        else:
            return ''
    elif hasattr(file, 'read'):
        return file.read()
    elif callable(file):
        return file()
    else:
        raise ValueError(f"Invalid file type: {type(file)}")


def write_file(file, content, append=False):
    """
    Write to a file flexibly. Will create parent directories if they don't exist.

    :param file:  file as object, path string, .read()-able, or callable.
    :param content: string content to write
    :param append: whether to append to an existing file
    :return: None
    """
    if isinstance(file, (str, pathlib.Path)):
        file_path = pathlib.Path(file)
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True)
        with open(file, 'a+' if append else 'w') as f:
            f.write(content)
    elif hasattr(file, 'write'):
        file.write(content)
    elif callable(file):
        file(content)
    else:
        raise ValueError(f"Invalid file type: {type(file)}")


def find_in_cache(cache, key):
    """
    Try to find a json-serializable item in a cache file.

    :param cache: file as object, path string, .read()-able, or callable.
    :param key: item to look for on the top level of the json file
    :return: bool whether item was found, value associated with the item
    """
    cache = json.loads(read_file(cache) or '{}')
    key = json.dumps(key)
    if key in cache:
        return True, cache[key]
    else:
        return False, None

def save_to_cache(cache, key, item):
    cache_map = json.loads(read_file(cache) or '{}')
    key = json.dumps(key)
    cache_map[key] = item
    write_file(cache, json.dumps(cache_map, indent=2))