
import regex


list_items = r'[0-9]+\.(.*)'
'''1. Lorem ipsum'''

list_label_items = r'[0-9]+\.([^:]+): (.*)'
'''1. Foo: Lorem ipsum'''

list_label_paren_items = r'[0-9]+\.([^:]+): ([^(]+)\(([^)]+)\)'
'''1. Foo: Lorem ipsum (bar bat baz)'''

label_items = r'([^:]+):(.*)'
'''Foo: lorem ipsum'''


def parse(text, pattern: str | list[str]):
    """
    Parse text using one or more regexes.

    Passing multiple regexes allows for cascading fallbacks; the first regex to match will be used to parse and return output.

    :param text: string of text
    :param pattern: regex string
    :return: list of strings matching regex groups
    """
    if isinstance(pattern, str):
        patterns = [pattern]
    else:
        patterns = pattern
    empty_match = False
    for pattern in patterns:
        try:
            if isinstance(pattern, str):
                found = regex.findall(pattern, text)
            else:
                found = pattern.findall(text)
            if found:
                return [
                    item.strip() if isinstance(item, str)
                    else [subitem.strip() for subitem in item]
                    for item in found
                ]
            else:
                empty_match = True
        except regex.error:
            continue
    if empty_match:
        return []
    raise ValueError(f'No pattern matched {text}')