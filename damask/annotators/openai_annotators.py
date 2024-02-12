from .base import Annotator
from typing import Dict, Any
import os
from dotenv import load_dotenv

from litellm import embedding, completion
from .base import Annotator
from typing import Dict, Any

load_dotenv()

"""
create a .env file in the root directory of the project
with the following contents:
    OPENAI_API_KEY=<your openai api key>
    OPENAI_ORG=<your openai organization id>
"""


class EmbeddingAnnotator(Annotator):
    def __init__(self, model):
        self.model = model

    def annotate(self, text: str) -> Dict[str, Any]:
        response = self.model.embedding.create(input=text)
        embedding = response.data[0].embedding
        return {"embedding": embedding}


class ChatCompletionAnnotator(Annotator):
    def __init__(
        self, model, system_prompt, user_prompt="{text}", tag="chat_completion"
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.tag = tag

    def annotate(self, text) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt.format(text=text)},
        ]
        response = self.model.completion(messages=messages, stream=True)
        chat_completion = next(response).choices[0].delta.content or ""
        return {self.tag: chat_completion}
