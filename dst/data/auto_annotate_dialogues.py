import tqdm

from promptium.prompt import prompt
import promptium.parse as parse

import dataclasses as dc
import pickle
import random
import pathlib
import multiprocessing
import time
import regex
import json
import itertools
import shutil


def load_dialogues(path: str) -> list['Dialogue']:
    objs = []
    with open(path, 'rb') as file:
        for scenario, dialogues in pickle.load(file).items():
            for dialogue in dialogues:
                turns = []
                for speaker, turn in dialogue:
                    if 'info' not in ''.join(c.lower() for c in speaker if c.isalpha()):
                        if turns:
                            previous_speaker, previous_turn = turns[-1]
                            if previous_speaker == speaker:
                                turns[-1] = (speaker, f'{previous_turn} {turn}')
                                continue
                        turns.append((speaker, turn))

                speakers = set(list(zip(*turns))[0])
                if turns:
                    obj = Dialogue(scenario, speakers)
                    for index, (speaker, turn) in enumerate(turns):
                        obj.turns.append(Turn(turn, obj, index, speaker))
                    objs.append(obj)
    return objs



@dc.dataclass
class Dialogue:
    scenario: str
    speakers: set[str] = dc.field(default_factory=set)
    turns: list['Turn'] = dc.field(default_factory=list)



@dc.dataclass
class Turn:
    text: str
    dialogue: Dialogue
    index: int
    speaker: str = None
    example: 'Example' = None

    @property
    def listener(self):
        if self.index == 0 and len(self.dialogue.turns) > 1:
            return self.dialogue.turns[1].speaker
        elif self.index > 0:
            return self.dialogue.turns[self.index-1].speaker
        else:
            return 'Listener'

    @property
    def window(self):
        return self.dialogue.turns[:self.index + 1]

    @property
    def context(self):
        return self.dialogue.turns[:self.index]

    @property
    def previous(self):
        if self.index > 0:
            return self.dialogue.turns[self.index-1]
        else:
            return None

    @property
    def next(self):
        if self.index < len(self.dialogue.turns) - 1:
            return self.dialogue.turns[self.index+1]
        else:
            return None


def parse_question_answer_pairs(text, speaker, listener):
    degenerations = 0
    speaker_key = ''.join(c.lower() for c in speaker if c.isalnum())
    listener_key = ''.join(c.lower() for c in listener if c.isalnum())
    pairs = []
    lines = text.split('\n')
    pair = []
    for line in lines:
        if not line.strip():
            pair = []
        parts = line.split(':', 1)
        if len(parts) == 2:
            person, turn = parts
            person = regex.sub(r'[0-9]+\. ', '', person)
            person_key = ''.join(c.lower() for c in person if c.isalnum())
            if not pair and person_key in (speaker_key, listener_key):
                if not turn.strip() or turn.strip()[-1] != '?':
                    degenerations += 1
                elif person_key == speaker_key:
                    pair.append((speaker, turn.strip()))
                else:
                    pair.append((listener, turn.strip()))
            elif pair and person_key in (speaker_key, listener_key):
                turnkey = ''.join(c.lower() for c in turn if c.isalnum())
                if turnkey == 'unknown':
                    turn = 'Unknown.'
                previous_key = ''.join(c.lower() for c in pair[0][0] if c.isalnum())
                if person_key == previous_key:
                    degenerations += 1
                elif person_key == speaker_key:
                    pair.append((speaker, turn.strip()))
                    pairs.append(pair)
                    pair = []
                else:
                    pair.append((listener, turn.strip()))
                    pairs.append(pair)
                    pair = []
            else:
                degenerations += 1
    qas = [(s1, q, s2, a) for (s1, q), (s2, a) in pairs]
    return qas, degenerations

def requests_and_informs(
    qa_quads: list[tuple[str, str, str, str]],
    speaker: str,
) -> tuple[list[str], list[tuple[str, str]]]:
    requests = []
    informs = []
    for s1, q, s2, a in qa_quads:
        if s1 == speaker and a == 'Unknown.':
            requests.append(q)
        elif s2 == speaker and a != 'Unknown.':
            informs.append((q, a))
    return requests, informs

def camel_case_to_text(s):
    new = ''
    for i, ch in enumerate(s):
        if i > 0 and s[i-1].islower() and ch.isupper():
            new += ' ' + ch.lower()
        else:
            new += ch
    return new

def snake_case_to_text(s):
    return ' '.join(s.split('_'))

def variable_to_slot_name(variable:str):
    slot_name = camel_case_to_text(variable)
    slot_name = snake_case_to_text(slot_name)
    return slot_name

@dc.dataclass
class Slot:
    example: 'Example'
    name: str = None
    value: str = None
    question: str = None
    answer: str = None
    description: str = None
    alternatives: list[str] = dc.field(default_factory=list)
    is_categorical: bool = None

@dc.dataclass
class Example:
    turn: Turn
    requests: list[str] = dc.field(default_factory=list)
    answered: list[tuple[str, str]] = dc.field(default_factory=list)
    unanswered: list[str] = dc.field(default_factory=list)
    informs: list[tuple[str, str]] = dc.field(default_factory=list)
    slots: dict[str, 'Slot'] = dc.field(default_factory=dict)

    def __init__(self,
        turn: Turn,
        requests: list[str] = None,
        answered: list[tuple[str, str]] = None,
        unanswered: list[str] = None,
        informs: list[tuple[str, str]] = None,
        slots: dict[str, 'Slot'] = None
    ):
        self.turn = turn
        self.requests = requests or []
        self.answered = answered or []
        self.unanswered = unanswered or []
        self.informs = informs or []
        self.slots = slots or {}
        self.turn.example = self

    @property
    def previous(self):
        if self.turn.previous:
            return self.turn.previous.example
        else:
            return None

    @property
    def next(self):
        if self.turn.next:
            return self.turn.next.example
        else:
            return None

    @property
    def answered_and_unanswered_slots(self):
        answered_slots = []
        unanswered_slots = []
        answered = set(list(zip(*self.answered))[0]) if self.answered else set()
        unanswered = set(self.unanswered)
        for slot in self.slots.values():
            if slot.question in answered:
                answered_slots.append(slot)
            elif slot.question in unanswered:
                unanswered_slots.append(slot)
        return answered_slots, unanswered_slots

    @property
    def answered_questions(self):
        return {q: a for q, a in self.informs}

    @property
    def new_answered(self):
        return [q for q, a in self.answered if not self.previous or q not in self.previous.requests]

    @property
    def new_unanswered(self):
        return [q for q in self.unanswered if not self.previous or q not in self.previous.requests]

    @property
    def new_informs(self):
        previous_requests = set(self.previous.requests) if self.previous else set()
        return [(q, a) for q, a in self.informs if not self.previous or q not in previous_requests]

    @property
    def carried_quesions(self):
        previous_requests = set(self.previous.requests)
        return [
            q for q in list(zip(*self.answered))[0] + self.unanswered
            if self.previous and q in previous_requests
        ]

    @property
    def slot_questions(self):
        return {slot.question: slot for slot in self.slots.values()}

    @property
    def carried_slot_questions(self):
        if not self.previous:
            return {}
        else:
            previous_requests = set(self.previous.requests)
            previous_slot_questions = self.previous.slot_questions
            return {
                q: previous_slot_questions[q] for q in self.carried_quesions
                if q in previous_slot_questions and q in previous_requests
            }

    @property
    def carried_unslotted_questions(self):
        if not self.previous:
            return []
        else:
            previous_slot_questions = self.previous.slot_questions
            return [
                q for q in self.carried_quesions
                if q not in previous_slot_questions
            ]

    @property
    def carried_slotted_and_answered(self):
        if not self.previous:
            return {}
        else:
            previous_slot_questions = self.previous.slot_questions
            return {
                q: (previous_slot_questions[q], a) for q, a in self.answered
                if q in previous_slot_questions
            }

    @prompt(model='gpt-4', temperature=0.1)
    def gen_qa_pairs(self, llm=None):
        """
        Two people, {speaker} and {listener}, are having a dialogue in which the following was just said:

        {context}{turn}

        Please break down and summarize all the information in what {speaker} just said into as many question-answer pairs as you can. Each question-answer pair should be short, specific, and focus on only one piece of information or value.

        For information {speaker} shared, use the question-answer pair format:

        {listener}: <question>
        {speaker}: <answer>

        For information {speaker} requested or indicated not knowing, use the answer "Unknown." in a question-answer pair format like:

        {speaker}: <question>
        {listener}: Unknown.


        {previous}
        """
        previous_informs = self.previous.informs if self.previous else []
        generated = llm.generate(
            speaker=self.turn.speaker,
            listener=self.turn.listener,
            context=(
                f'{self.turn.listener}: {self.turn.previous.text}\n\n'
                if self.turn.previous else ''
            ),
            turn = f'{self.turn.speaker}: {self.turn.text}',
            previous=(
                '\n\n'.join(
                    f'{self.turn.listener}: {q}\n{self.turn.speaker}: {a}'
                    for q, a in self.answered
                ) + '\n\n' if self.answered else ''
            )
        )
        qas, degens = parse_question_answer_pairs(
            generated, self.turn.speaker, self.turn.listener
        )
        self.requests, self.informs = requests_and_informs(qas, self.turn.speaker)
        answered = [(q, a) for q, a in self.answered if a != 'Unknown.']
        self.informs = answered + self.informs
        return self.requests, self.informs

    @prompt(model='gpt-4', temperature=0.1)
    def gen_qa_answers(self, llm=None):
        """
        Two people, {speaker} and {listener}, are having a dialogue in which the following was just said:

        {context}{turn}

        Please identify the information or values {speaker} gave as short answers to the following questions (use the answer "Unknown." if the question is not answered by {speaker} in the dialogue):

        {questions}
        """
        if not self.previous or not self.previous.requests:
            return self.answered
        questions = '\n\n'.join(
            f'{self.turn.listener}: {q}\n{self.turn.speaker}:' for q in self.previous.requests
        ) + ('\n\n' if len(self.previous.requests) > 1 else '')
        generated = llm.generate(
            speaker=self.turn.speaker,
            listener=self.turn.listener,
            context=(
                f'{self.turn.previous.speaker}: {self.turn.previous.text}\n\n'
                if self.turn.previous else ''
            ),
            turn = f'{self.turn.speaker}: {self.turn.text}',
            questions=questions
        )
        def try_parse_answers_only(generated, questions):
            answers = [line.split(':')[1].strip() for line in generated.split('\n') if ':' in line]
            if len(questions) == len(answers):
                qas = [(q, a) for q, a in zip(questions, answers)]
                return qas
            else:
                return []
        answer_qas = try_parse_answers_only(generated, self.previous.requests)
        if answer_qas:
            generated = '\n\n'.join(
                f'{self.turn.listener}: {q}\n{self.turn.speaker}: {a}'
                for q, a in answer_qas
            )
        elif len(self.previous.requests) == 1:
            generated = questions + ' ' + generated
        qas, degens = parse_question_answer_pairs(
            generated, self.turn.speaker, self.turn.listener
        )
        _, self.answered = requests_and_informs(qas, self.turn.speaker)
        self.unanswered = [q for q in self.previous.requests if q not in {q for q, a in self.answered}]
        return self.unanswered, self.answered

    @prompt(temperature=0.1)
    def gen_slot_names(self, llm=None):
        """
        {qas}

        Translate each question above into variable names. Each label should be very short, usually one or two words, but specific to the details of the question. Write each question before translating it into a variable name, in the format:

        <question> -> <variable name>


        """
        questions = self.new_informs + [(q, 'Unknown.') for q in self.requests]
        if questions:
            generated = llm.generate(
                qas = '\n\n'.join(q for q, a in questions)
            )
            def question_arrow_slot(generated):
                slots = {}
                keyify = lambda s: ''.join(c for c in s if c.isalnum()).lower()
                quesiton_keys = {keyify(q): (q, a) for q, a in questions}
                lines = generated.split('\n')
                previous = None
                for line in (line.strip() for line in lines if line.strip()):
                    question, slot = line.split('->', 1) if '->' in line else ('', '')
                    question = question.strip()
                    slot = slot.strip()
                    if question and slot:
                        question = regex.sub(r'\(?[0-9]+[.)] ', '', question)
                        question = keyify(question)
                        slot = slot.strip()
                        if question in quesiton_keys:
                            slots[slot] = quesiton_keys[question]
                    elif previous and line:
                        question, slot = previous, line
                        if '->' in slot:
                            slot = slot.split('->', 1)[1].strip()
                        slots[slot] = quesiton_keys[question]
                        previous = None
                    else:
                        key = keyify(line)
                        if key in quesiton_keys:
                            previous = key
                        else:
                            previous = None
                return slots
            slots = question_arrow_slot(generated)
        else:
            slots = {}
        carried = {}
        requests = {}
        informs = {}
        for q, (slot, a) in self.carried_slotted_and_answered.items():
            self.slots[slot.name] = dc.replace(slot, answer=a, value=None)
            carried[slot.name] = q
        for slot_name, (q, a) in slots.items():
            if a == 'Unknown.':
                slot = Slot(example=self, name=slot_name, question=q, answer=a, value='?')
                requests[slot_name] = q
            else:
                slot = Slot(example=self, name=slot_name, question=q, answer=a)
                informs[slot_name] = q
            self.slots[slot_name] = slot
            if q in self.new_answered:
                self.previous.slots[slot_name] = dc.replace(slot, example=self.previous, answer='Unknown.', value='?')
        return carried, informs, requests

    @prompt(temperature=0.1)
    def gen_slot_value(self, llm=None):
        """
        {qas}

        Translate each answer to the above questions into a value for the corresponding variable. Values should be short, usually one word, very short phrase, number, span, category, score, boolean, list, or other value. Copy each answer before translating it into a value, in the format:

        Question: <question>
        Variable: <variable>
        Answer: <answer>
        Value: <value>


        """
        slots = {slot.name: slot for slot in self.slots.values() if slot.value is None}
        if not slots:
            return slots
        generated = llm.generate(
            qas = '\n\n'.join(
                f'Question: {slot.question}\nVariable: {slot.name}\nAnswer: {slot.answer}'
                for slot in slots.values()
            )
        )
        outputs = parse.parse(generated, parse.label_items)
        current_slot = None
        for label, value in outputs:
            if label == 'Variable':
                current_slot = value.strip()
            elif label == 'Value' and current_slot in slots:
                slots[current_slot].value = value.strip()
                if (
                    value.startswith('"') and value.endswith('"')
                    or value.startswith("'") and value.endswith("'")
                    or value.startswith('[') and value.endswith(']')
                    or value.startswith('{') and value.endswith('}')
                    or value.startswith('(') and value.endswith(')')
                ):
                    value = value[1:-1]
                slots[current_slot].value = value
                current_slot = None
        result = {slot.name: slot.value for slot in slots.values()}
        return result


    @prompt(temperature=0.1)
    def gen_slot_description(self, llm=None):
        """
        {questions_and_slots}

        For each Info Type above, write a comma-separated list of all Possible Values (if there are many Possible Values, write ", etc." after a few examples), and a short phrase as a description for each Info Type. Use the format:

        Info Type: <info type>
        Possible Values: <value 1>, <value 2>, <value 3>
        Description: <phrase>


        """
        newline = '\n'
        if not self.next:
            slots = self.slots
        else:
            next_requests = set(self.next.requests)
            slots = {
                slot.name: slot for slot in self.slots.values()
                if slot.name not in self.next.slots
                and slot.question not in next_requests
            }
        if not slots:
            return {}
        generated = llm.generate(
            questions_and_slots = '\n\n'.join(
                f'Question: {slot.question}\n'
                f'Info Type: {variable_to_slot_name(slot.name)}\n'
                f'{"Example Value: " + slot.value + newline if slot.value not in ("?", None) else ""}'
                f'Description:'
                for slot in slots.values()
            )
        )
        lines = generated.split('\n')
        slots_values_descriptions = []
        info_type = None
        possible_values = None
        description = None
        for line in lines:
            if ':' in line:
                label, value = line.split(':', 1)
                label = label.strip()
                value = value.strip()
                if label == 'Info Type':
                    info_type = value
                elif label == 'Possible Values':
                    possible_values = value
                elif label == 'Description':
                    description = value
                if info_type and possible_values and description:
                    slots_values_descriptions.append((info_type, possible_values, description))
                    info_type = None
                    possible_values = None
                    description = None
        slot_map = {variable_to_slot_name(slot.name): slot for slot in self.slots.values()}
        for slot_name, possible_values, description in slots_values_descriptions:
            slot = slot_map.get(slot_name)
            if slot is None:
                continue
            slot.alternatives = set(x.strip() for x in possible_values.split(','))
            if slot.value not in ('?', None):
                slot.alternatives.add(slot.value)
            slot.description = description
            if self.previous and slot_name in self.previous.slots:
                if self.previous.slots[slot_name].question == slot.question:
                    self.previous.slots[slot_name].alternatives = slot.alternatives
                    self.previous.slots[slot_name].description = slot.description

def gen_dst_data(
    dialogues: list[Dialogue],
    num_examples=None,
    num_dialogues=None,
    seed=None,
    display=1,
    just_qa_pairs=False,
    include_slot_description=True,
    **ops
):
    if seed:
        random.seed(seed)
    if num_dialogues:
        dialogues = random.sample(dialogues, num_dialogues)
        if num_examples:
            dialogues = [dialogue.turns[:num_examples] for dialogue in dialogues]
            examples = [[Example(turn) for turn in turns] for turns in dialogues]
        else:
            examples = [[Example(turn) for turn in dialogue.turns] for dialogue in dialogues]
    else:
        examples = [[Example(turn) for turn in dialogue.turns] for dialogue in dialogues]
    num_generated_examples = 0
    for i, dialogue in tqdm.tqdm(enumerate(examples)):
        previously = None
        for j, example in enumerate(dialogue):
            example.gen_qa_answers(**ops)
            example.gen_qa_pairs(**ops)
            if not just_qa_pairs:
                example.gen_slot_names(**ops)
                example.gen_slot_value(**ops)
            for x in ([previously] if previously else []) + ([example] if not example.next else []):
                if not just_qa_pairs and include_slot_description:
                    x.gen_slot_description(**ops)
                # display_example(x)
            previously = example
            num_generated_examples += 1
        if display and i % display == 0:
            print(f'Generated {num_generated_examples} examples for {i+1} dialogues')
    return examples

def display_example(x:Example):
    print('#' * 80)
    if x.previous:
        print(f'{x.previous.turn.speaker}: {x.previous.turn.text}')
    print(f'{x.turn.speaker}: {x.turn.text}\n')
    for slot in x.slots.values():
        print(f'{slot.name}: {slot.value}')

def select_dialogues_round_1(dialogues: list[Dialogue]):
    scenarios = {}
    for dialogue in dialogues:
        scenarios.setdefault(dialogue.scenario, []).append(dialogue)
    selected = []
    for scenario, scenario_dialouges in scenarios.items():
        selected.append(scenario_dialouges[0])
    print(f'Selected {len(selected)} dialogues out of {len(dialogues)}')
    return selected

def select_dialogues_round_2(dialogues: list[Dialogue]):
    scenarios = {}
    for dialogue in dialogues:
        scenarios.setdefault(dialogue.scenario, []).append(dialogue)
    selected = []
    for scenario, scenario_dialouges in scenarios.items():
        selected.extend(scenario_dialouges[1:5])
    print(f'Selected {len(selected)} dialogues out of {len(dialogues)}')
    return selected


def multiprocess_job(dialogues: list[Dialogue], just_qa_pairs, i, folder, api_key_location=None):
    print('Job', i, 'started with', len(dialogues), 'dialogues')
    folder = pathlib.Path('llm_cache')/folder
    cache_folder = folder/f'cache_{i}'
    cache_folder.mkdir(exist_ok=True, parents=True)
    for cache_file in folder.glob('*.gen.json'):
        with open(cache_file) as f:
            cache = json.load(f)
        with open(cache_folder/cache_file.name, 'w') as f:
            json.dump(cache, f)
    with open(cache_folder/f'dialogues.pkl', 'wb') as f:
        pickle.dump(dialogues, f)
    api_key_args = {}
    with open(api_key_location) as f:
        org, key = [x.strip() for x in f.readlines()]
        api_key_args.update(api_key=key, api_org=org)
    examples = gen_dst_data(
        dialogues, just_qa_pairs=just_qa_pairs, cache_folder=str(cache_folder), **api_key_args
    )
    with open(cache_folder/f'examples.pkl', 'wb') as f:
        pickle.dump(examples, f)

def multiprocess_dst_data(
    dialogues: list[Dialogue],
    just_qa_pairs: bool,
    num_processes: int,
    folder_name: str,
    api_key_locations: list[str] = None
):
    batch_size = (len(dialogues) + num_processes - 1) // num_processes
    batches = [dialogues[i:i+batch_size] for i in range(0, len(dialogues), batch_size)]
    with multiprocessing.Pool(num_processes) as pool:
        pool.starmap(
            multiprocess_job, zip(
                batches,
                itertools.repeat(just_qa_pairs),
                range(len(batches)),
                itertools.repeat(folder_name),
                itertools.cycle(api_key_locations or [None])
            ))

def coalesce_multiprocessed(folder, remove_subfolders=False):
    folder = pathlib.Path('llm_cache')/folder
    dialogues = []
    examples = []
    caches = {}
    for cache in folder.glob('*.gen.json'):
        caches[cache.name] = json.loads(cache.read_text())
    for cache_folder in folder.glob('cache_*'):
        dialogues_file = cache_folder/'dialogues.pkl'
        if dialogues_file.exists():
            with open(dialogues_file, 'rb') as f:
                dialogues.extend(pickle.load(f))
        examples_path = cache_folder/'examples.pkl'
        if examples_path.exists():
            with open(examples_path, 'rb') as f:
                examples.extend(pickle.load(f))
        for cache_file in cache_folder.glob('*.gen.json'):
            cache = json.loads(cache_file.read_text())
            caches.setdefault(cache_file.name, {}).update(cache)
    with open(folder/'dialogues.pkl', 'wb') as f:
        pickle.dump(dialogues, f)
    with open(folder/'examples.pkl', 'wb') as f:
        pickle.dump(examples, f)
    for cache_file, cache in caches.items():
        with open(folder/cache_file, 'w') as f:
            json.dump(cache, f)
    if remove_subfolders:
        for cache_folder in folder.glob('cache_*'):
            shutil.rmtree(cache_folder)

if __name__ == '__main__':
    ...





