
import json

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize


def parse(gen):
    jsons = []
    stack = []
    for i, ch in enumerate(gen):
        if ch == '{':
            stack.append(i)
        elif stack and ch == '}':
            j = stack.pop()
            jsons.append(gen[j:i+1])
    structs = []
    for jso in jsons:
        try:
            structs.append(json.loads(jso))
        except json.JSONDecodeError:
            continue
    return structs

def flatten(struct):
    if isinstance(struct, dict):
        items = {
            flatten(k): flatten(v) for k, v in struct.items()
        }
        items = {k: v for k, v in items.items() if k is not None and v is not None}
        struct = {}
        for key, val in items.items():
            if isinstance(val, dict):
                struct.update({key+' '+k: v for k, v in val.items()})
            else:
                struct[key] = val
        return struct
    elif isinstance(struct, list):
        items = [flatten(item) for item in struct]
        items = [item for item in items if item is not None]
        struct = {}
        for item in items:
            if isinstance(item, dict):
                struct.update(item)
        items = ', '.join([x for x in items if isinstance(x, str)])
        if items:
            struct[''] = items
        return struct
    elif isinstance(struct, (float, int)):
        return str(struct)
    elif isinstance(struct, bool):
        return 'yes' if struct else 'no'
    elif isinstance(struct, type(None)):
        return 'N/A'
    else:
        return clean_naming(struct)

def camel_case_to_text(s):
    if ' ' in s.strip():
        return s
    for i, ch in enumerate(s):
        if i > 0 and s[i-1].islower() and ch.isupper():
            s = s[:i] + ' ' + s[i:]
    return s

def snake_case_to_text(s):
    if ' ' in s.strip():
        return s
    return ' '.join(s.split('_'))

def text_to_alpha(s):
    return ''.join([c for c in s if c.isalpha() or c == ' ']).strip()

def is_nonsense(s):
    return text_to_alpha(s.strip().lower()) in {
        'variable', 'var', 'speaker', 'text', 'turn'
    }

def clean_naming(s):
    s = camel_case_to_text(s)
    s = snake_case_to_text(s)
    if is_nonsense(s):
        return None
    return s.strip()

def clean_example(dialogue, extractions, turn_idx=None):
    noise = {"slots": ["Dialogue", "message"], "values": []}
    # split the dialogue into sentences and split the sentences into a ["SPEAKER", "SENTENCE"] format
    if isinstance(dialogue, str):
        dsplit = []
        for line in enumerate(dialogue.split('\n')):
            dsplit.append(line[1].split(':', 1))
    else:
        dsplit = dialogue
    ex_cleaned = {}  # store the cleaned key & values
    for slot in extractions:
        # remove the noise slots such as "Dialogue"
        if (slot in noise["slots"]) or (extractions[slot] in noise["values"]):
            continue
        # remove the repeated sentences
        is_rep = False
        for sentence in dsplit:
            if not len(sentence) == 2: continue
            if extractions[slot].strip().casefold() == sentence[1].strip().casefold():
                is_rep = True
            # remove partially repeated sentences
            sent_split = sent_tokenize(sentence[1].strip().casefold())
            for ext in sent_tokenize(extractions[slot]):
                if ext in sent_split and len(ext) > 8:
                    is_rep = True
        if is_rep:
            continue
        else:
            ex_cleaned[slot] = extractions[slot]
    if turn_idx is not None:
        # remove references to the speakers themselves
        context = dialogue[max(0, turn_idx-2):turn_idx+1]
        lower_alpha_only = lambda s: ''.join([c for c in s if c.isalpha() or c == ' ']).strip().lower()
        context_key = lower_alpha_only('\n'.join(t for s, t in context))
        speaker_keys = {lower_alpha_only(s) for s, t in dialogue}
        for speaker in list(speaker_keys):
            speaker_keys.update((f'{speaker} name', f'name of {speaker}'))
        for slot, value in list(ex_cleaned.items()):
            slot_key = lower_alpha_only(slot)
            value_key = lower_alpha_only(value)
            if slot_key in context_key:
                continue
            elif slot_key in speaker_keys:
                del ex_cleaned[slot]
            elif slot_key == 'name' and value_key in speaker_keys:
                del ex_cleaned[slot]
    return ex_cleaned


def get_dialogue(prompt):
    prompt = eval(prompt)
    lines = prompt.split('\n')
    start = lines.index('Dialogue:') + 1
    return '\n'.join([line for line in lines[start:] if line.strip()])
