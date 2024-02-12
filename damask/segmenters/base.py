from abc import ABC, abstractmethod, ABCMeta
from functools import wraps
from typing import List, Dict, Any
import re
import uuid
from functools import wraps
from damask.models.segmenter import Segmenter
import re
from typing import Dict, Generator, Any

"""
This module contains the base classes for segmenters.  
Segmenters take a text as input and generate a list of annotations.
Annotations consist of a start index, an end index, and metadata associated with that segment.

Classes:
    Segmenter: The interface for segmenters.
    SentenceSegmenter: A segmenter that segments a damask text into sentences.
    WordSegmenter: A segmenter that segments a damask text into words.
    ChunkSegmenter: A segmenter that segments a damask text into chunks of a given size in characters.
"""


class SentenceSegmenter(Segmenter):
    """
    A segmenter that segments a damask text into sentences.
    """

    # Modify the regular expression to include line breaks
    sentence_endings_pattern = re.compile(r"(?<=\S)[.!?](?=\s|$)|\n")

    def segment(self, text: str) -> Generator[Dict[str, Any], None, None]:
        """
        Segments the given text into sentences based on punctuation marks and line breaks.

        Parameters:
        - text (str): The input text to be segmented into sentences.

        Yields:
        - Generator[Dict[str, Any], None, None]: A generator of dictionaries, each representing a sentence with its start and end indices and metadata.
        """
        start = 0
        for match in self.sentence_endings_pattern.finditer(text):
            end = match.end()
            yield {"start": start, "end": end, "metadata": {}}
            start = end + 1 if text[end : end + 1].isspace() else end
        if start < len(text):
            yield {"start": start, "end": len(text), "metadata": {}}


class WordSegmenter(Segmenter):
    # Compile the regular expression once
    token_pattern = re.compile(r"\b[a-zA-Z0-9]+\b|[.!?]")

    def segment(self, text: str) -> Generator[Dict[str, Any], None, None]:
        """
        Segments the given text into words by identifying word boundaries.

        Parameters:
        - text (str): The input text to be segmented into words.

        Yields:
        - Generator[Dict[str, Any], None, None]: A generator of dictionaries, each representing a word with its start and end indices and metadata.
        """
        for match in self.token_pattern.finditer(text):
            yield {"start": match.start(), "end": match.end(), "metadata": {}}


class ChunkSegmenter(Segmenter):
    """
    A segmenter that segments a damask text into chunks of a given size in characters.
    """

    def __init__(self, chunk_size: int):
        """
        Initializes the ChunkSegmenter with a specific chunk size.

        Parameters:
        - chunk_size (int): The size of each chunk in characters.
        """
        super().__init__()
        self.chunk_size = chunk_size

    def segment(self, text: str) -> Generator[Dict[str, Any], None, None]:
        """
        Segments the given text into chunks of a predefined size using a generator.

        Parameters:
        - text (str): The input text to be segmented into chunks.

        Yields:
        - Generator[Dict[str, Any], None, None]: A generator of dictionaries, each representing a chunk
        with its start and end indices and metadata.
        """
        for start in range(0, len(text), self.chunk_size):
            end = min(start + self.chunk_size, len(text))
            yield {"start": start, "end": end, "metadata": {}}
