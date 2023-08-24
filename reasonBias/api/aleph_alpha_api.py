from pathlib import Path
from typing import List, Optional

from aleph_alpha_client import AlephAlphaClient

from reasonBias.api.languageModelInterface import LanguageModelInterface


class AlephAlphaLM(LanguageModelInterface):

    def __init__(self, key: str = None, limit: Optional[int] = None, dry_run: bool = False, model_name="luminous-base", max_tokens=50,
                 suppressControl=False, int100query=False):
        super().__init__(key, limit=limit, dry_run=dry_run)
        self.max_tokens = max_tokens

        if model_name is None:
            raise ValueError("No model name provided.")
        self.model_name = model_name

        if isinstance(key, Path):
            with open(key / "aleph_alpha", "r") as f:
                token = f.readline()
        elif isinstance(key, str):
            token = key
        else:
            raise ValueError("unknown parameter type.")

        self.client = AlephAlphaClient(
            host="https://api.aleph-alpha.com",
            token=token
        )

        supports_control = self.model_name == "luminous-supreme-control"
        if supports_control:
            if suppressControl:
                self._query_type = "basic"
            else:
                self._query_type = "control"
        else:
            self._query_type = "basic"

        if int100query:
            if self._query_type == "basic":
                raise RuntimeError("int100query requires 'control'.")
            self._query_type = "int100query"

        self._limit_variance_factor = 5 # 10

    def do_query(self, query_text: str) -> str:
        result = self.client.complete(
            self.model_name,
            query_text,
            maximum_tokens=self.max_tokens,
            temperature=0.0,
            #top_k=0,
            #top_p=0,
            presence_penalty=0,
            frequency_penalty=0
        )

        return result['completions'][0]['completion']

    def do_query_variance(self, query_text: str, temperature: float, num_samples: int, **kwargs) -> List[str]:
        if self._query_type == "basic":
            pass # query text is left unmodified
        elif self._query_type == "control":
            query_text = f"### Instruction:\n{query_text}\n\n### Response:\n"
        elif self._query_type == "int100query":
            query_text = f"### Instruction:\nGive an answer the following question.\nReply with a single number on a scale of 1 to 100.\n\n### Input:\n{query_text}\n\n### Response:\n"
        else:
            raise RuntimeError(f"Invalid query type ({self._query_type}).")

        responses = self.client.complete(
            self.model_name,
            query_text,
            maximum_tokens=self.max_tokens,
            temperature=temperature*0.5,  # temperature is between [0..1], while all others recomend [0..2]. So we rescale here.
            #top_k=50,
            #top_p=1.0,
            frequency_penalty=0,
            presence_penalty=0,
            n=num_samples,
            **kwargs
        )
        response_texts = []
        for response in responses['completions']:
            response_texts.append(response['completion'])
        return response_texts
