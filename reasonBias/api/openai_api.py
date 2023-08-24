from pathlib import Path
from typing import Optional, List

import openai

from reasonBias.api.languageModelInterface import LanguageModelInterface


class OpenAIBase(LanguageModelInterface):

    def __init__(self, key: str = None, limit: Optional[int] = 2000, dry_run: bool = False, model_name=None):
        # complete & embedding: limit for free account is 20/min, 60/min for paid in the first 48h and 3500 afterwards.
        # chat: limit for free account is 20/min, 60/min for paid in the first 48h and 3500 afterwards.
        super().__init__(key, limit=limit, dry_run=dry_run)

        if model_name is None:
            raise ValueError("No model name provided.")
        self.model_name = model_name

        if isinstance(key, Path):
            with open(key / "openai", "r") as f:
                openai.api_key = f.readline()  # os.getenv("OPENAI_API_KEY")
        elif isinstance(key, str):
            openai.api_key = key
        else:
            raise ValueError("Unknown parameter type.")

        if model_name == "gpt-3.5-turbo":
            self._limit_variance_factor = 30.0
        elif model_name == "gpt-4":
            self._limit_variance_factor = 57.0


class OpenAILM(OpenAIBase):

    def __init__(self, key: str = None, limit: Optional[int] = 3000, dry_run: bool = False, model_name="text-davinci-003", max_tokens=50, is_chat=False):
        super().__init__(key, limit, dry_run, model_name)
        self.max_tokens = max_tokens
        self.is_chat = is_chat

    def _extract_answer(self, answer):
        if self.is_chat:
            return answer["message"]["content"]
        else:
            return answer["text"]

    def do_query(self, query_text: str) -> str:
        if self.is_chat:
            messages = [{
                "role": "user",
                "content": query_text
            }]
            response = openai.ChatCompletion.create(
                # engine=self.model_name,
                model=self.model_name,
                messages=messages,
                temperature=0,
                max_tokens=self.max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
        else:
            response = openai.Completion.create(
                #engine=self.model_name,
                model=self.model_name,
                prompt=query_text,
                temperature=0,
                max_tokens=self.max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
        return self._extract_answer(response['choices'][0])

    def do_query_variance(self, query_text: str, temperature: float, num_samples: int, **kwargs) -> List[str]:
        if self.is_chat:
            messages = [{
                "role": "user",
                "content": query_text
            }]
            responses = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_tokens,
                #top_p=1,
                # top_k=50, # openai does not support top-k sampling ...
                frequency_penalty=0,
                presence_penalty=0,
                n=num_samples,
                **kwargs
            )
        else:
            responses = openai.Completion.create(
                model=self.model_name,
                prompt=query_text,
                temperature=temperature,
                max_tokens=self.max_tokens,
                #top_p=1,
                #top_k=50, # openai does not support top-k sampling ...
                frequency_penalty=0,
                presence_penalty=0,
                n=num_samples,
                **kwargs
            )
        response_texts = []
        for response in responses['choices']:
            response_texts.append(self._extract_answer(response))
        return response_texts

    #{"role": "user", "content": "Who won the world series in 2020?"},


class OpenAILMEmbedding(OpenAIBase):

    def __init__(self, key: str = None, limit: Optional[int] = 3000, dry_run: bool = False, model_name="text-embedding-ada-002"):
        super().__init__(key, limit, dry_run, model_name)

    def do_get_embedding(self, query_text: str) -> list:
        response = openai.Embedding.create(
            input=query_text,
            engine=self.model_name
        )
        embedding = response["data"][0]["embedding"]
        return embedding

    def do_get_embeddings_batched(self, query_texts: List[str]) -> list:
        response = openai.Embedding.create(
            input=query_texts,
            engine=self.model_name
        )

        embeddings = [""]*len(query_texts)
        for answer in response["data"]:
            embeddings[answer.index] = answer.embedding
        return embeddings
