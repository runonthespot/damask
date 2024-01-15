from abc import ABC, abstractmethod
from functools import wraps
import uuid
from typing import Union, Generator, List, Dict, Any
from collections.abc import Iterable


class Segmenter(ABC):
    def __init__(self):
        self.uuid_map = {}
        self._initialized = True

    def get_segment_id(self, segment_text: str) -> uuid.UUID:
        if segment_text not in self.uuid_map:
            self.uuid_map[segment_text] = uuid.uuid4()
        return self.uuid_map[segment_text]

    def _segment_with_uuids(func):
        @wraps(func)
        def wrapper(
            self, text: str
        ) -> Union[Generator[Dict[str, Any], None, None], List[Dict[str, Any]]]:
            result = func(self, text)
            if isinstance(result, Iterable) and not isinstance(result, (str, bytes)):
                for annotation in result:
                    segment_text = text[annotation["start"] : annotation["end"]]
                    if "metadata" not in annotation:
                        annotation["metadata"] = {}
                    annotation["metadata"]["id"] = str(
                        self.get_segment_id(segment_text)
                    )
                    print(annotation)
                    if isinstance(result, Generator):
                        yield annotation
                    else:
                        return [annotation for annotation in result]
            else:
                raise TypeError(
                    "The segment function must return an iterable (list or generator)."
                )

        return wrapper

    @abstractmethod
    @_segment_with_uuids
    def segment(
        self, text: str
    ) -> Union[Generator[Dict[str, Any], None, None], List[Dict[str, Any]]]:
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.segment = Segmenter._segment_with_uuids(cls.segment)

    @abstractmethod
    def segment(self, text: str) -> List[Dict[str, Any]]:
        pass
