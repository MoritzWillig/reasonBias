from abc import ABC, abstractmethod
from typing import Optional, Any, List
import time


class LanguageModelInterface:

    def __init__(self, key: str = None, limit: Optional[int] = None, dry_run: bool = False):
        """
        :param key:
        :param limit: limit interface to N request per minute
        :param dry_run:
        """
        self.dry_run = dry_run
        self._limit = limit
        self._min_request_time = None if self._limit is None else 60.0 / self._limit
        self._last_request_time = None

        self._limit_variance_factor = None

        self._max_variance_batch_size = 100

    def _await_limit(self, factor=1.0):
        if self._limit is None:
            return
        if self._last_request_time is None:
            self._last_request_time = time.perf_counter()
            return

        current_time = time.perf_counter()
        elapsed_time = current_time - self._last_request_time
        remaining_time = (self._min_request_time * factor) - elapsed_time
        if remaining_time > 0.0:
            time.sleep(remaining_time)
        self._last_request_time = time.perf_counter()

    def query(self, query_text: str, log_info: Any = "") -> Optional[str]:
        print(f"[querying{'' if log_info=='' else ' | '}{log_info}]", query_text)
        if self.dry_run:
            return None
        self._await_limit()
        answer = self.do_query(query_text)
        return answer

    def query_variance(self, query_text: str, temperature: int, num_samples: int, log_info: Any = "", **kwargs) -> Optional[List[str]]:
        print(f"[querying{'' if log_info=='' else ' | '}{log_info}]", query_text)
        if self.dry_run:
            return None
        self._await_limit(factor=self._limit_variance_factor)

        all_answers = []
        remaining_num_samples = num_samples
        if remaining_num_samples != 0:
            next_num_samples = min(remaining_num_samples, self._max_variance_batch_size)
            answers = self.do_query_variance(query_text, temperature, next_num_samples, **kwargs)
            remaining_num_samples -= next_num_samples

            all_answers.extend(answers)
        return all_answers

    def query_embedding(self, query_text: str, log_info: Any = "") -> Optional[list]:
        print(f"[querying{'' if log_info=='' else ' | '}{log_info}]", query_text)
        if self.dry_run:
            return None
        self._await_limit()
        answer = self.do_get_embedding(query_text)
        return answer

    def get_embeddings_batched(self, query_texts: List[str], log_info: Any = "") -> Optional[list]:
        print(f"[querying batch{'' if log_info == '' else ' | '}{log_info}]")
        for q in query_texts:
            print(">", q)

        if self.dry_run:
            return None
        self._await_limit()
        answer = self.do_get_embeddings_batched(query_texts)
        return answer

    def do_query(self, query_text: str) -> str:
        raise NotImplementedError()

    def do_query_variance(self, query_text: str, temperature: float, num_samples: int, **kwargs) -> List[str]:
        raise NotImplementedError()

    def do_get_embedding(self, query_text: str) -> List:
        raise NotImplementedError()

    def do_get_embeddings_batched(self, query_texts: List[str]) -> List[str]:
        raise NotImplementedError()
