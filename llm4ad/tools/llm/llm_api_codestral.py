import traceback
from mistralai import Mistral
from ...base import LLM

class MistralApi(LLM):
    def __init__(self, keys: str, model: str = "codestral-latest", timeout=60, **kwargs):
        super().__init__(**kwargs)

        # Initialize client
        self._client = Mistral(api_key=keys)
        self.model_name = model
        self._timeout = timeout
        self._kwargs = kwargs
        self._api_key = keys  # stored only for info/logging

    def draw_sample(self, prompt: str, suffix: str = "", *args, **kwargs) -> str:
        try:
            print(f"[INFO] Calling Mistral model: {self.model_name} with key {self._api_key[:8]}...")

            response = self._client.chat.complete(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=kwargs.get("temperature", 1)
                )
            return response.choices[0].message.content

        except Exception:
            print(f"[ERROR] Mistral API call failed:\n{traceback.format_exc()}")
            return "API_FAILED"
