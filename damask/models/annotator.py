from abc import ABC, abstractmethod
from typing import Dict, Any


class Annotator(ABC):
    @abstractmethod
    def annotate(self, text: str) -> Dict[str, Any]:
        pass
