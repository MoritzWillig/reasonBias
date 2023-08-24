import importlib
from typing import Type
#from reasonBias.api.opt_api import OptLM
#from reasonBias.api.openai_api import OpenAILM
#from reasonBias.api.aleph_alpha_api import AlephAlphaLM

from reasonBias.api.languageModelInterface import LanguageModelInterface

model_config = {
    "davinci-002": ["openai_api", "OpenAILM", {"model_name": "text-davinci-002"}],
    "davinci-003": ["openai_api", "OpenAILM", {"model_name": "text-davinci-003"}], # openai recomends to use gpt-3.5-turbo instead.
    "gpt-3.5-turbo": ["openai_api", "OpenAILM", {"model_name": "gpt-3.5-turbo", "is_chat": True, "limit": 190}], # XXX
    "gpt-4": ["openai_api", "OpenAILM", {"model_name": "gpt-4", "is_chat": True, "limit": 190}],

    #"chat_gpt": ["openai_api", "OpenAILM", {"model_name": "gpt-4", "is_chat": True}],

    "davinci-001-embedding": ["openai_api", "OpenAILMEmbedding", {"model_name": "text-similarity-davinci-001"}],
    "ada-002-embedding": ["openai_api", "OpenAILMEmbedding", {"model_name": "text-embedding-ada-002"}],

    "luminous-base": ["aleph_alpha_api", "AlephAlphaLM", {"model_name": "luminous-base"}],
    "luminous-extended": ["aleph_alpha_api", "AlephAlphaLM", {"model_name": "luminous-extended"}],
    "luminous-supreme": ["aleph_alpha_api", "AlephAlphaLM", {"model_name": "luminous-supreme"}],
    "luminous-supreme-control": ["aleph_alpha_api", "AlephAlphaLM", {"model_name": "luminous-supreme-control"}],
    "luminous-supreme-control_nocontrol": ["aleph_alpha_api", "AlephAlphaLM", {"model_name": "luminous-supreme-control", "suppressControl": True}],
    "luminous-supreme-control_int100query": ["aleph_alpha_api", "AlephAlphaLM", {"model_name": "luminous-supreme-control", "int100query": True, "max_tokens":10}],

    "opt-30b": ["opt_api", "OptLM", {"model_name": "facebook/opt-30b"}],
    "opt-iml-max-30b": ["opt_api", "OptLM", {"model_name": "facebook/opt-iml-max-30b"}],
    "opt-66b": ["opt_api", "OptLM", {"model_name": "facebook/opt-66b"}],

    "t5-small-lm-adapt": ["opt_api", "OptLM", {"model_name": "google/t5-small-lm-adapt"}],
}

_model_cache = {}


def get_lm_by_name(name) -> Type[LanguageModelInterface]:
    global _model_cache
    if name in _model_cache:
        return _model_cache[name]

    module_name, lm_class_name, s_kwargs = model_config[name]

    module = importlib.import_module("."+module_name, __package__)
    lm_class = module.__dict__[lm_class_name]

    lm_func = lambda *args, **kwargs: lm_class(*args, **{**s_kwargs, **kwargs})
    _model_cache[name] = lm_func
    return lm_func
