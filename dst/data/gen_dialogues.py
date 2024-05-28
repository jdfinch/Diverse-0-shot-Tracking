import regex

from promptium.prompt import prompt
import promptium.parse as parse
import promptium.gpt as gptapi
import dst.data.parse as dst_parse
import pickle
import json
import sentence_transformers as st
import sentence_transformers.util as stu
import pathlib

def gen_lotsa_tasks(n, **ops):
    file_path = pathlib.Path('llm_cache/lotsa_tasks.json')
    model = st.SentenceTransformer('all-MiniLM-L6-v2')
    if file_path.exists():
        with open(file_path) as f:
            tasks = json.load(f)
    else:
        tasks = []
    while len(tasks) < n:
        updated = []
        if tasks:
            ops.update(gen_recache='lotsa_tasks')
        new_tasks = gen_tasks(100, **ops)
        if not isinstance(new_tasks, list):
            continue
        all_tasks = tasks + new_tasks
        embeddings = model.encode(all_tasks, convert_to_tensor=True)
        groups = stu.community_detection(
            embeddings, threshold=0.80, min_community_size=1,
        )
        for i, group in enumerate(reversed(groups)):
            print(f'Group {i}:')
            print('\n'.join([f'  {all_tasks[idx]}' for idx in group]))
            selection = all_tasks[sorted(group)[-1]]
            updated.append(selection)
        with open(file_path, 'w') as f:
            json.dump(updated, f)
        tasks = updated


@prompt
def gen_tasks(n, generated=None):
    """
    List {n} diverse examples of everyday tasks that require talking to another person. Format each list item like:

    N. <Role of person 1> talks to <role of person 2> in order to <task goal>


    """
    scenarios = parse.parse(generated, parse.list_items)
    return scenarios

@prompt
def gen_ontology(task, idx=0, generated=None):
    """
    List examples of as many different types of information as you can that would be shared during the dialogue scenario: {task}


    """
    return generated

@prompt
def gen_dialogue(task, ontology, generated=None):
    """
    Dialogue Scenario:
    {task}

    Information Types:
    {ontology}

    Write a dialogue for the above Dialogue Scenario. Include specific examples of the Information Types above being shared and implied throughout the conversation. Make up actual names/values when specific information examples are shared.


    """
    return parse.parse(generated, parse.label_items)

@prompt
def gen_extract(ontology, dialogue, generated=None):
    """
    Extract as many variables as you can from the Dialogue, formatted in JSON (use "?" if a value is being requested), like

    {
        "<variable name>": <value>,
        ...
    }

    Dialogue:
    {dialogue}


    """
    return dst_parse.parse(generated), dialogue


def gen_pipeline(**ops):
    path = pathlib.Path('llm_cache', 'gpt10k')
    tasks = json.loads((path / 'tasks.json').read_text())
    print(f'Generating {len(tasks)} tasks')
    dialogues = []
    for task in tasks:
        ontology = gen_ontology(task, **ops)
        dialogue = gen_dialogue(task, ontology, **ops)
        dialogues.append(dialogue)
        # for i in range(2, len(dialogue)):
        #     history = dialogue[max(i - 3, 0):i]
        #     context = '\n'.join([f'{s}: {t}' for s, t in history])
            # gen_extract(ontology, context, **ops)
    return dialogues

import multiprocessing as mp

def gen_sub_pipeline(tasks, n_per_task, k, cache_folder='llm_cache', ops=None):
    if ops is None:
        ops = {}
    ops.update(dict(
        cache_folder=f'{cache_folder}/{k}th_proc'
    ))
    for j, task in enumerate(tasks):
        for m in range(n_per_task):
            ontology = gen_ontology(task, m, gen_recache=True, **ops)
            dialogue = gen_dialogue(task, ontology, **ops)
            for i in range(2, len(dialogue)):
                history = dialogue[max(i - 3, 0):i]
                context = '\n'.join([f'{s}: {t}' for s, t in history])
                gen_extract(ontology, context, **ops)
            print(f'Process {k} completed {j+1}/{len(tasks)} tasks')
            print(
                '   ',
                f'{gptapi.tokens:,} tokens used in {gptapi.runtime()/60:.1f} min',
                flush=True
            )

def gen_multi_pipeline(tasks, n_per_task, procs, cache_folder='llm_cache', **ops):
    tasks = [tasks[i::procs] for i in range(procs)]
    with mp.Pool(procs) as pool:
        pool.starmap(
            gen_sub_pipeline,
            [(tasks[i], n_per_task, i, cache_folder, ops) for i in range(procs)]
        )



if __name__ == '__main__':
    gen_lotsa_tasks(1000)
    gen_pipeline()


