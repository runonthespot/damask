from abc import ABC, abstractmethod


class TextRepository(ABC):
    @abstractmethod
    def get_text(self, text_id: str):
        pass

    @abstractmethod
    def save_text(self, text_id: str, content: str):
        pass


class InMemoryTextRepository(TextRepository):
    def __init__(self):
        self.storage = {}

    def get_text(self, text_id: str):
        return self.storage.get(text_id)

    def save_text(self, text_id: str, content: str):
        self.storage[text_id] = content
