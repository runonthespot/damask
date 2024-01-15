from openai import OpenAI
from .base import Annotator
from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

"""
create a .env file in the root directory of the project
with the following contents:
    OPENAI_API_KEY=<your openai api key>
    OPENAI_ORG=<your openai organization id>
"""


class EmbeddingAnnotator(Annotator):
    _api_key = os.getenv("OPENAI_API_KEY")
    _org = os.getenv("OPENAI_ORG")

    def __init__(self):
        self.client = OpenAI(
            api_key=self._api_key,
            organization=self._org,
        )

    def annotate(self, text: str) -> Dict[str, Any]:
        response = self.client.embeddings.create(
            input=text, model="text-embedding-ada-002"
        )
        embedding = response.data[0].embedding
        return {"embedding": embedding}


class ChatCompletionAnnotator(Annotator):
    _api_key = os.getenv("OPENAI_API_KEY")
    _org = os.getenv("OPENAI_ORG")

    def __init__(self, system_prompt, user_prompt="{text}"):
        self.client = OpenAI(
            api_key=self._api_key,
            organization=self._org,
        )
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    def annotate(self, text) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model="gpt-4-1106-preview",  # or another model you prefer
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt.format(text=text)},
            ],
        )
        chat_completion = response.choices[0].message.content
        return {"chat_completion": chat_completion}
