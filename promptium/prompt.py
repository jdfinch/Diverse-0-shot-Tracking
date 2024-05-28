
import functools
import inspect
import pathlib
import re
import textwrap
import json
import typing as T

from promptium.gpt import gpt
import promptium.disk as io
from promptium.crashless import crashless

def default(option, default): return default if option is None else option

llm_param = 'llm'
generated_param = 'generated'

F = T.TypeVar('F')


class LLM:

    def __init__(
        self,
        fn: F | 'LLM',
        model=None,
        temperature=None,
        prompt=None,
        prompt_sep=None,
        log=None,
        cache=None,
        recache=None,
        gen_cache=None,
        gen_recache=None,
        cache_folder=None,
        debug=False,
        api_key=None,
        api_org=None,
    ):
        if isinstance(fn, LLM):
            self._copy(fn)
        else:
            self.fn = fn
            self.signature = inspect.signature(fn)
        self.api_key = default(api_key, getattr(self, 'api_key', api_key))
        self.api_org = default(api_org, getattr(self, 'api_org', api_org))
        self.model, self.temperature = self._init_model(model, temperature)
        self.prompt_sep = default(prompt_sep, getattr(self, 'prompt_sep', None))
        self.template = self._init_prompt(fn, prompt)
        prompt_params, template = self._init_prompt_params(self.template)
        self.prompt_params = prompt_params
        self.template = template
        self.log = self._init_log(log)
        self.cache = self._init_cache(cache, 'cache')
        self.recache = self._init_cache(recache, 'recache')
        self.gen_cache = self._init_cache(
            gen_cache, 'gen_cache', suffix='.gen'
        )
        self.gen_recache = self._init_cache(
            gen_recache, 'gen_recache', suffix='.gen'
        )
        self.cache_folder = pathlib.Path(default(
            cache_folder, getattr(self, 'cache_folder', 'llm_cache')
        ))
        self.debug = default(debug, getattr(self, 'debug', debug))
        self.input = None
        self.prompt = None
        self.generated = None
        self.output = None

    def _copy(self, other: 'LLM'):
        self.fn = other.fn
        self.signature = other.signature
        self.model = other.model
        self.prompt_sep = other.prompt_sep
        self.template = other.template
        self.prompt_params = other.prompt_params
        self.log = other.log
        self.cache = other.cache
        self.recache = other.recache
        self.gen_cache = other.gen_cache
        self.gen_recache = other.gen_recache
        self.cache_folder = other.cache_folder
        self.debug = other.debug
        self.input = other.input
        self.generated = other.generated

    def _init_model(self, model, temperature=None):
        if temperature is not None:
            temperaturearg = dict(temperature=temperature)
        else:
            temperaturearg = {}
        if model is None:
            bound_gpt = functools.partial(
                gpt, api_key=self.api_key, api_org=self.api_org, **temperaturearg
            )
            model = getattr(self, 'model', bound_gpt)
        if isinstance(model, str):
            model = functools.partial(
                gpt, model=model, api_key=self.api_key, api_org=self.api_org, **temperaturearg
            )
        elif not callable(model):
            raise ValueError(
                f'generate argument {model} must be a OpenAI model ID or Callable[[str], str]')
        return model, temperature

    def _init_prompt(self, fn, prompt):
        if (
            prompt is None and not hasattr(self, 'template')
            and getattr(fn, '__doc__')
        ):
            lines = fn.__doc__.split('\n')
            if not lines[0].strip():
                if not lines[-1].strip():
                    lines = lines[1:-1]
                else:
                    lines = lines[1:]
            last_documentation_index = len(lines)
            for i, trailing_line in enumerate(reversed(lines)):
                if not (
                    trailing_line.strip().startswith(':param') or
                    trailing_line.strip().startswith(':return:')
                ):
                    if i > 0:
                        last_documentation_index = len(lines) - i - 1
                    break
            lines = lines[:last_documentation_index]
            prompt = textwrap.dedent('\n'.join(lines))
            return prompt
        else:
            prompt = getattr(self, 'template', prompt)
            return prompt

    def _init_llm_prompt(self, llm_prompt, split_llm_prompt):
        return default(llm_prompt or split_llm_prompt, getattr(self, 'llm_template', ''))

    def _init_prompt_params(self, prompt, which='template'):
        if prompt is None:
            if hasattr(self, 'prompt_params'):
                return self.prompt_params, getattr(self, which)
            else:
                return [], getattr(self, which)
        param_matches = list(re.finditer('{[a-zA-Z0-9_]*}', prompt))
        param_map = []
        params = []
        for i, param_match in enumerate(param_matches):
            param_string = param_match.group()
            param_string = param_string[1:-1]
            if not param_string:
                param_string = str(i)
            if param_string not in params:
                params.append(param_string)
            param_map.append(f'{{{param_string}}}')
        for param_match, param_string in zip(
            reversed(param_matches), reversed(param_map)
        ):
            start, end = param_match.regs[0]
            prompt = prompt[:start] + param_string + prompt[end:]
        return params, prompt

    def _init_log(self, log):
        if log is True:
            log = f'{self.fn.__name__}.log'
        return default(log, getattr(self, 'log', log))

    def _init_cache(self, cache, attr_name, suffix=''):
        if cache is True:
            cache = f'{self.fn.__name__}{suffix}.json'
        cache = default(cache, getattr(self, attr_name, cache))
        if attr_name == 'gen_cache' and cache is None:
            cache = f'{self.fn.__name__}{suffix}.json'
        return cache

    def generate(
        self, *args: tuple[str], prompt: str=None, **kwargs: dict[str, str]
    ):
        self.prompt = self.fill_prompt(*args, prompt=prompt, **kwargs)
        prompt = self.prompt
        llm_prompt = None
        if self.prompt_sep:
            splitter = self.prompt.rfind(self.prompt_sep)
            if splitter != -1:
                prompt = self.prompt[:splitter]
                llm_prompt = self.prompt[splitter + len(self.prompt_sep):]
        prompt_in_cache = False
        if self.gen_cache and not self.gen_recache:
            prompt_in_cache, self.generated = io.find_in_cache(
                self.cache_folder/self.gen_cache, self.prompt
            )
        if not prompt_in_cache:
            api_key_arg = dict(api_key=self.api_key) if self.api_key else {}
            api_org_arg = dict(api_org=self.api_org) if self.api_org else {}
            self.generated = (llm_prompt or '') + self.model(
                prompt=prompt, assistant_prompt=llm_prompt, **api_key_arg, **api_org_arg
            )
            cache = self.gen_cache or self.gen_recache
            if cache:
                io.save_to_cache(
                    self.cache_folder/cache, self.prompt, self.generated
                )
        return self.generated

    def __call__(
        self,
        *args,
        log=None,
        cache=None,
        recache=None,
        gen_cache=None,
        gen_recache=None,
        cache_folder=None,
        debug=None,
        api_key=None,
        api_org=None,
        prompt=None,
        **kwargs
    ):
        wrapper = LLM(
            self, log=log, cache=cache, recache=recache,
            gen_cache=gen_cache, gen_recache=gen_recache,
            cache_folder=cache_folder, debug=debug, api_key=api_key, api_org=api_org,
        )
        wrapper.input = self._bind_input(self.signature, args, kwargs)
        input_in_cache = False
        if wrapper.cache and not wrapper.recache:
            input_in_cache, cached = io.find_in_cache(
               wrapper.cache_folder/wrapper.cache, wrapper.input
            )
            if cached:
                wrapper.output = cached
        if not input_in_cache:
            wrapper.output = wrapper._call(*args, **kwargs)
            if wrapper.cache or wrapper.recache:
                cache = wrapper.recache or wrapper.cache
                io.save_to_cache(
                    wrapper.cache_folder/cache, wrapper.input, wrapper.output
                )
        if wrapper.log:
            wrapper._log(
                wrapper.log,
                wrapper.input, wrapper.prompt,
                wrapper.generated, wrapper.output
            )
        return wrapper.output

    def __get__(self, instance, owner):
        return functools.partial(self.__call__, instance)

    def _bind_input(self, signature, args, kwargs):
        binding = signature.bind(*args, **kwargs)
        binding.apply_defaults()
        arguments = binding.arguments
        arguments = {
            k: v for k, v in arguments.items()
            if k not in (llm_param, generated_param)
        }
        return arguments

    def _call(self, *args, **kwargs):
        output = None
        if generated_param in self.signature.parameters:
            generated, output = None, '' # noqa
            with crashless(crashing_is_allowed=(not self.debug)):
                generated = self.generate()
                kwargs[generated_param] = generated
                with crashless(crashing_is_allowed=(not self.debug)):
                    output = self.fn(*args, **kwargs)
        elif llm_param in self.signature.parameters:
            kwargs[llm_param] = self
            with crashless(crashing_is_allowed=(not self.debug)):
                output = self.fn(*args, **kwargs)
        else:
            with crashless(crashing_is_allowed=(not self.debug)):
                generated = self.generate()
        return output

    def fill_prompt(self, *prompt_args, prompt=None, **prompt_kwargs):
        prompt = self.template if prompt is None else prompt
        if prompt_args or prompt_kwargs:
            args, kwargs = prompt_args, prompt_kwargs
        else:
            args, kwargs = [], self.input
        arg_map = {}
        for param, arg in zip(self.prompt_params, args):
            arg_map[param] = arg
        arg_map.update(kwargs)
        for i, (param, arg) in enumerate(arg_map.items()):
            if arg is None:
                prompt = prompt.replace(f' {{{param}}}', '')
                prompt = prompt.replace(f' {{{i}}}', '')
                prompt = prompt.replace(f'{{{param}}} ', '')
                prompt = prompt.replace(f'{{{i}}} ', '')
                prompt = prompt.replace(f'{{{param}}}', '')
                prompt = prompt.replace(f'{{{i}}}', '')
            else:
                prompt = prompt.replace(f'{{{param}}}', str(arg))
                prompt = prompt.replace(f'{{{i}}}', str(arg))
        return prompt

    def _log(self, log, input, prompt, generated, output):
        log_sep = '-' * 80
        log_header = f'{self.fn.__name__}:'
        try:
            log_input = f'input: {json.dumps(input)}'
        except Exception:
            log_input = f'input: <Could not parse>'
        log_text = f'{prompt}<<<{generated}>>>'
        try:
            log_output = f'output: {json.dumps(output)}'
        except Exception:
            log_output = f'output: <Could not parse>'
        log_content = '\n'.join([
            log_sep,
            log_header, '',
            textwrap.indent(log_input, '    '), '',
            textwrap.indent(log_text, '    '), '',
            textwrap.indent(log_output, '    '), '',
            log_sep
        ])
        log_content = textwrap.dedent(log_content)
        io.write_file(log, log_content, append=True)


def prompt(
    fn=None, model=None, temperature=None, log=None, cache=None, recache=None, cache_folder=None,
    gen_cache=None, gen_recache=None, debug=None, prompt_sep='||', api_key=None, api_org=None,
) -> LLM | F:
    if fn is None:
        return functools.partial(
            prompt, model=model, temperature=temperature,
            log=log, cache=cache, recache=recache, cache_folder=cache_folder,
            gen_cache=gen_cache, gen_recache=gen_recache, debug=debug,
            prompt_sep=prompt_sep, api_key=api_key, api_org=api_org
        )
    api = LLM(
        fn=fn,
        model=model,
        temperature=temperature,
        prompt=None,
        prompt_sep=prompt_sep,
        log=log,
        cache=cache,
        recache=recache,
        gen_cache=gen_cache,
        gen_recache=gen_recache,
        debug=debug,
        cache_folder=cache_folder,
        api_key=api_key,
        api_org=api_org,
    )
    return api


