from typing import Union, Literal
import pdb 
import logging
import time
from abc import ABC, abstractmethod
from typing import Callable

import openai
from typing_extensions import override

NUM_LLM_RETRIES = 10

LOG: logging.Logger = logging.getLogger(__name__)

class LLM(ABC):
    def __init__(self, model: str, api_key: str) -> None:
        self.model: str = model
        self.api_key: str | None = api_key

    @abstractmethod
    def query(self, prompt: str) -> str:
        """
        Abstract method to query an LLM with a given prompt and return the response.

        Args:
            prompt (str): The prompt to send to the LLM.

        Returns:
            str: The response from the LLM.
        """
        pass

    def query_with_system_prompt(self, system_prompt: str, prompt: str) -> str:
        """
        Abstract method to query an LLM with a given prompt and system prompt and return the response.

        Args:
            system prompt (str): The system prompt to send to the LLM.
            prompt (str): The prompt to send to the LLM.

        Returns:
            str: The response from the LLM.
        """
        return self.query(system_prompt + "\n" + prompt)

    def _query_with_retries(
        self,
        func: Callable[..., str],
        *args: str,
        retries: int = NUM_LLM_RETRIES,
        backoff_factor: float = 0.5,
    ) -> str:
        last_exception = None
        for retry in range(retries):
            try:
                return func(*args)
            except Exception as exception:
                last_exception = exception
                sleep_time = backoff_factor * (2**retry)
                time.sleep(sleep_time)
                LOG.debug(
                    f"LLM Query failed with error: {exception}. Sleeping for {sleep_time} seconds..."
                )
                print(f"LLM Query failed with error: {exception}. Sleeping for {sleep_time} seconds...")
        raise RuntimeError(
            f"Unable to query LLM after {retries} retries: {last_exception}"
        )

    def query_with_retries(self, prompt: str, stop_seqs=[], max_tokens=1024, num_outputs=1) -> str:
        return self._query_with_retries(self.query, prompt, stop_seqs, max_tokens, num_outputs)

    def query_with_system_prompt_with_retries(
        self, system_prompt: str, prompt: str, stop_seqs=[], max_tokens=1024, num_outputs=1) -> str:
        return self._query_with_retries(
            self.query_with_system_prompt, system_prompt, prompt, stop_seqs, max_tokens, num_outputs)
    
class OPENAI(LLM):
    """Accessing OPENAI or VLLM model"""

    def __init__(self, model_name: str, api_key: str, base_url=None) -> None:
        super().__init__(model_name, api_key)
        if base_url is not None:
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url) 
        else:
            self.client = openai.OpenAI(api_key=api_key)  
        self.name = model_name 

    @override
    def query(self, prompt: str, stop_seqs = None, max_tokens=1024, num_outputs=1) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            stop=stop_seqs,
            n=num_outputs 
        )
        if num_outputs==1:
            return response.choices[0].message.content
        return [r.message.content for r in response.choices]
     
    @override
    def query_with_system_prompt(self, system_prompt: str, prompt: str, stop_seqs = None, max_tokens=1024, num_outputs=1) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            stop=stop_seqs,
            n=num_outputs 
        )
        if num_outputs==1:
            return response.choices[0].message.content
        return [r.message.content for r in response.choices]
        