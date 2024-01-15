from typing import List, Dict, Any, Callable, Tuple, Optional
import json


class Annotation:
    def __init__(self, start: int, end: int, metadata: Dict[str, Any]):
        self.start = start
        self.end = end
        self.metadata = metadata

    def __str__(self):
        pretty_metadata = json.dumps(self.metadata, indent=4, sort_keys=True)
        return f"Annotation(start={self.start}, end={self.end}, length={self.length}, metadata={pretty_metadata})"

    def __repr__(self):
        return self.__str__()

    def adjust_bounds(
        self, transform: Callable[[int, int], Tuple[int, int]], text_length: int
    ) -> None:
        new_start, new_end = transform(self.start, self.end)
        self.start = max(
            0, min(new_start, text_length)
        )  # Ensure start is within bounds
        self.end = max(
            self.start, min(new_end, text_length)
        )  # Ensure end is within bounds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "length": self.length,
            "metadata": self.metadata,
        }

    @property
    def length(self) -> int:
        return (self.end - self.start) + 1

    def overlaps_with(self, other) -> bool:
        return self.start <= other.end and self.end >= other.start
