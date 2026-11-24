from langchain_openai import ChatOpenAI
from typing import Optional

from ragkb.interfaces.base_generator import BaseGenerator


class LLMGenerator(BaseGenerator):
    def __init__(self, model_name:str,api_key : str, base_url :Optional[str] = None,temperature:float =0):
        self.llm = ChatOpenAI(
            model = model_name,
            api_key = api_key,
            base_url = base_url,
            temperature = temperature
        )

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("prompt 为空")

        response = self.llm.invoke(prompt)

        return response.content