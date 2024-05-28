"""
Module for generating text using chatgpt.

Create the file ~/.keys/openai with two lines:
1. openai organization id
2. openai api key
(get these in your openai user profile page)
"""

import sys
import time
import json
import http.client
import pathlib
import random

import openai

keypath = pathlib.Path.home() / '.keys' / 'openai'

try:
    with open(keypath) as f:
        org, key = [x.strip() for x in f.readlines() if x]
except FileNotFoundError:
    org, key = None, None
openai.organization = org
openai.api_key = key

ada = 'text-ada-001'
davinci = 'text-davinci-003'
chatgpt = 'gpt-3.5-turbo'
gpt4 = 'gpt-4'

default = object()

wait_time = 0
tokens = 0
calls = 0
gpt4_tokens = 0
gpt4_calls = 0
start_time = -1
runtime = lambda: time.time() - start_time

num_gpt4_calls = 0


def gpt(
    prompt,
    model=chatgpt,
    n=1,
    max_tokens=2000,
    temperature=1.0,
    include_prompt=False,
    file=None,
    assistant_prompt='',
    report_usage=10,
    api_key=None,
    api_org=None,
):
    """
    Generate text using gpt, given a prompt.

    :param prompt: prompt input that model will append to via generation
    :param model: model to use-- ada is small, davinci is large
    :param n: number of generations to make using the prompt
    :param max_tokens: maximum number of tokens (prompt+generation) cutoff
    :param temperature: parameter controlling model output variance
    :param include_prompt: whether to include the prompt in outputs
    :param file: string name of file to append to
    :param assistant_prompt: starting generation for the assistant
    :param report_usage: whether to report usage
    :param api_key: openai api key
    :param api_org: openai organization id
    :return: list of strings representing outputs (generations wrapped in square braces [])
    """
    global wait_time
    global tokens
    global gpt4_tokens
    global calls
    global gpt4_calls
    global start_time
    if start_time < 0:
        start_time = time.time()
    old_key = None
    old_org = None
    if api_key is not None:
        old_key = openai.api_key
        openai.api_key = api_key
        openai.organization = None
    if api_org is not None:
        old_org = openai.organization
        openai.organization = api_org
    global num_gpt4_calls
    if 'gpt-3.5' in model or 'gpt-4' in model:
        if 'gpt-4' in model:
            ...
            # assert False, 'Gpt4 should already be finished for this run'
        # assert False, "We're not doing any calls right now!"
        initial_time = time.perf_counter()
        while True:
            if wait_time > 0:
                time_to_wait = min(30, wait_time)
                time.sleep(time_to_wait * 0.8 + time_to_wait * random.random() * 0.4)
            try:
                messages = [dict(role='user', content=prompt)]
                if assistant_prompt:
                    messages.append(dict(role='assistant', content=assistant_prompt))
                completions = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=n,
                )
                if 'gpt-4' in model:
                    gpt4_tokens += completions.usage.total_tokens
                    gpt4_calls += 1
                else:
                    tokens += completions.usage.total_tokens
                    calls += 1
                wait_time *= 0.8
                if report_usage:
                    if not isinstance(report_usage, int):
                        report_usage = 1
                    if 'gpt-4' in model and gpt4_calls % report_usage == 0:
                        print(
                            f'GPT-4: {gpt4_tokens} tokens in {runtime()/60:.2f}min '
                            f'({60 * gpt4_tokens / runtime():.2f} tokens/min)'
                        )
                    elif 'gpt-3' in model and calls % report_usage == 0:
                        print(
                            f'GPT-3.5: {tokens} tokens in {runtime()/60:.2f}min '
                            f'({60*tokens / runtime():.2f} tokens/min)'
                        )

                break
            except (
                openai.error.RateLimitError, openai.error.Timeout, http.client.RemoteDisconnected,
                openai.error.APIConnectionError, openai.error.ServiceUnavailableError, openai.error.APIError,
            ) as e:
                time_of_error = time.perf_counter()
                time_delta = time_of_error - initial_time
                initial_time = time_of_error
                print(f'Calling GPT resulted in an error after {time_delta:.2f}s. Error: {type(e)}. Retrying...', file=sys.stdout)
                wait_time = wait_time * 1.5 if wait_time > 0.1 else 0.2
        outputs = [choice.message.content for choice in completions.choices]
        display_outputs = [
            prompt + f'\n\n<{model}>[{output}]'
            for output in outputs
        ]
    else:
        initial_time = time.perf_counter()
        prompt = prompt + assistant_prompt
        while True:
            try:
                completions = openai.Completion.create(
                    engine=model,
                    temperature=temperature,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    n=n
                )
                wait_time -= 0.1
                break
            except (openai.error.RateLimitError, TimeoutError, http.client.RemoteDisconnected,) as e:
                time_of_error = time.perf_counter()
                time_delta = time_of_error - initial_time
                initial_time = time_of_error
                print(f'Calling GPT resulted in an error after {time_delta:.2f}s. Error: {type(e)}. Retrying...', file=sys.stderr)
                wait_time = wait_time * 2 if wait_time > 0 else 0.2
        outputs = [choice.turn for choice in completions.choices]
        display_outputs = [
            prompt + f'  <{model}>[{output}]'
            for output in outputs
        ]
    separator = '\n' +'_' * 80 + '\n\n'
    display_output = ''.join([o+separator for o in display_outputs])
    if isinstance(file, str):
        with open(file, 'a') as f:
            f.write(display_output)
    elif hasattr(file, 'write'):
        file.write(display_output)
    elif callable(file):
        file(display_output)
    if include_prompt:
        outputs = [prompt+output for output in outputs]
    if old_key is not None:
        openai.api_key = old_key
    if old_org is not None:
        openai.organization = old_org
    return outputs if len(outputs) >= 2 else outputs[0]


class Prompt:
    """
    A prompt for gpt, used by subclassing Prompt.

    Set string property `template` to define the prompt, using $0, $1, ... to define prompt args.

    Initialize the prompt like MyPrompt('foo', 'bar') to get a prompt object where the two prompt args are filled by 'foo' and 'bar' respectively.

    Override/define the `.parse(output: str)` method to parse output into an arbitrary output object.

    Override/define the `.model(prompt: str)` method to define a text generator that uses the prompt as input.
    """

    model = staticmethod(gpt)
    parse = None
    template = None
    inputs = ([], {})

    def __init__(self, *values, model=None, parse=None, **kwargs):
        self.inputs = list(values), kwargs
        self.prompt = self.template
        self.prompt = self.fill(*values, **kwargs)
        if model:
            self.model = model
        if parse:
            self.parse = parse

    def fill(self, *values, **kwargs):
        prompt = self.prompt
        for i, value in enumerate(values):
            prompt = prompt.replace(f'${i}', value)
        return prompt

    def generate(
        self,
        model=default,
        parser=default,
        logfile=None,
        cache=None,
        recache=None
    ):
        """
        Generate text using the prompt.

        :param model: a callable function that takes a string prompt as input and generates string text as output
        :param parser: a callable function that takes a string text outputted by the model and parses it into a JSON object.
        :param logfile: optionally, a string defining a logfile (use '{}' in the path string, and it will be replaced with the Prompt class name).
        :param cache: optionally, a string defining a JSON cache file (use '{}' and it will be replaced by the Prompt class name. If specified, inputs that have already been given to the model will be looked up in the cache, and the corresponding output returned without re-querying the model.
        :param recache: like cache, but recreates a new cache file (clearing any old one that exists with the same name)
        :return: a string of model output text, and an object outputted by the parser (or None, if no parser is specified)
        """
        if recache:
            cache = recache
        cached = []
        if isinstance(cache, str) and not recache:
            cache = cache.replace('{}', self.__class__.__name__)
            with open(cache) as f:
                cached = json.load(f)
            for item in cached:
                item_input = item['input']
                if item_input == self.inputs:
                    return item['generated'], item['output']
        model = self.model if model is default else model
        generated = model(self.prompt)
        if parser is default and self.parse:
            output = self.parse(generated)
        elif callable(parser):
            output = parser(generated)  # noqa
        else:
            output = None
        if logfile and not cache:
            logfile = logfile.replace('{}', self.__class__.__name__)
            with open(logfile, 'a+') as f:
                f.write('\n'.join((
                    f"\n{'_'*80}",
                    f"\n{self.__class__.__name__}: {self.__class__.__doc__}",
                    f"\n{self.prompt}{model}#![{generated}]",
                    f"\n{output}" if output else ''
                )))
        if isinstance(cache, str):
            with open(cache, 'w') as f:
                cached.append({
                    'input': self.inputs,
                    'prompt': self.prompt,
                    'generated': generated,
                    'output': output
                })
                json.dump(cached, f, indent=2)
        return generated, output

