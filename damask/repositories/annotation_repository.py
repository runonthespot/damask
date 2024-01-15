from abc import ABC, abstractmethod
from typing import Iterator
from damask.models.annotation import Annotation


class AnnotationRepository(ABC):
    @abstractmethod
    def get_annotations(self, text_id: str) -> Iterator[Annotation]:
        pass

    @abstractmethod
    def save_annotations(self, text_id: str, annotations: Iterator[Annotation]):
        pass


class InMemoryAnnotationRepository(AnnotationRepository):
    def __init__(self):
        self.storage = {}

    def get_annotations(self, text_id: str) -> Iterator[Annotation]:
        # This could be a generator if annotations are stored in a way that supports it
        for annotation in self.storage.get(text_id, []):
            yield annotation

    def save_annotations(self, text_id: str, annotations: Iterator[Annotation]):
        # Convert the iterator to a list for storage
        self.storage[text_id] = list(annotations)
