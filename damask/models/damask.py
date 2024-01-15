import logging
import threading
from typing import Dict, Any, List, Callable, Tuple
from damask.models.annotation import Annotation
from damask.segmenters import Segmenter
from damask.annotators import Annotator
from damask.models.annotation_set_view import AnnotationSetView
from concurrent.futures import ThreadPoolExecutor, as_completed
from tabulate import tabulate

from damask.processors import AnnotationProcessor


class Damask:
    """
    A class for storing text, with annotations.

    Args:
        text (str): The full text to be stored.

    Attributes:
        full_text (str): The full text to be stored.
        annotations (list): A list of dictionaries, each with a start index, end index, and metadata.

    Methods:
        display_annotation_set_as_table: Displays all annotation_sets with their types, contents, and additional metadata in a table format for easy inspection.
        annotate: Annotates a substring of the text with metadata.
        segment_text: Segments the text using a given TextProcessor function and stores the annotations.
        enrich_annotations: Enriches existing annotations of a specific type with additional metadata, given a TextProcessor function.
        get_annotation_sets: Retrieves a list of unique annotation_set types based on annotation metadata.
        get_texts_of_annotation_type: Retrieves the actual text of annotations of a specific annotation_set type.
        retrieve_annotated_text: Retrieves the annotated text with inline annotations formatted by the given Mustache template.
        filter_by_metadata: Filters annotations by metadata key-value pair.
    """

    def __init__(self, text):
        self.full_text = text
        self.annotations = []
        self.lock = threading.Lock()  # Lock for thread-safe operations

    def __str__(self):
        return self.full_text

    def __getitem__(self, key):
        # Delegate the indexing to the full_text attribute
        return self.full_text[key]

    def __getattr__(self, name):
        if name in self.get_annotation_sets():
            return AnnotationSetView(self, name)
        else:
            # Default behavior if attribute is not an annotation_set type
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

    def __iter__(self):
        if hasattr(self, "_iter_attr"):
            return self._iter_attr
        else:
            raise TypeError(f"'{type(self).__name__}' object is not iterable")

    def annotation_sets_as_table(self, annotation_type: str = None):
        """
        Displays annotation sets of a specific type, or all if no type is provided, in a table format for easy inspection.
        If a specific type is provided with no matches, an error is thrown.
        """
        table = []
        for annotation in self.annotations:
            if (
                annotation_type is not None
                and annotation.metadata.get("type") != annotation_type
            ):
                continue
            metadata_items = annotation.metadata.items()
            metadata_str = ", ".join(
                f"{key}: {value}" for key, value in metadata_items if key != "type"
            )
            table.append(
                [
                    annotation.metadata.get("type", "N/A"),
                    annotation.start,
                    annotation.end,
                    annotation.end - annotation.start,  # Assuming length is end - start
                    self.full_text[annotation.start : annotation.end],
                    metadata_str,
                ]
            )

        if annotation_type is not None and not table:
            raise ValueError(f"No annotations found for type '{annotation_type}'")

        headers = [
            "Annotation Set Type",
            "Start",
            "End",
            "Length",
            "Text",
            "Additional Metadata",
        ]
        return tabulate(table, headers, tablefmt="grid")

    def annotate(self, start: int, end: int, metadata: Dict[str, Any] = None) -> None:
        if metadata is None:
            metadata = {}
        print(
            f"Annotating: start={start}, end={end}, metadata={metadata}"
        )  # Debug print
        new_annotation = Annotation(start, end, metadata)
        self.annotations.append(new_annotation)

    def segment_text(self, segmenter: Segmenter, annotation_type: str) -> None:
        for new_segment in segmenter.segment(self.full_text):
            new_segment_metadata = new_segment.get("metadata", {})
            new_segment_metadata["type"] = annotation_type
            print(f"Segmenting with metadata: {new_segment_metadata}")  # Debug print
            self.annotate(
                new_segment["start"], new_segment["end"], new_segment_metadata
            )

    def apply_function_to_annotations(
        self, annotations: List[Annotation], func: Callable[[int, int], Tuple[int, int]]
    ) -> List[Annotation]:
        return [
            Annotation(func(annotation.start, annotation.end), annotation.metadata)
            for annotation in annotations
        ]

    def enrich_annotations(
        self,
        enricher: Annotator,
        annotation_type: str,
        parallel: bool = False,
        workers: int = 10,
    ) -> None:
        def enrich_annotation(annotation: Annotation):
            if annotation.metadata.get("type") == annotation_type:
                text_segment = self.full_text[annotation.start : annotation.end]
                enriched_metadata = enricher.annotate(text_segment)
                with self.lock:
                    annotation.metadata.update(enriched_metadata)

        if parallel:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(enrich_annotation, ann)
                    for ann in self.annotations
                    if ann.metadata.get("type") == annotation_type
                ]
                for future in as_completed(futures):
                    future.result()  # You can handle exceptions here if needed
        else:
            for annotation in self.annotations:
                enrich_annotation(annotation)

    def get_annotation_sets(self):
        """
        Retrieves a list of unique annotation set types based on annotation metadata.

        Returns:
            list: List of unique annotation set types.
        """
        return list(
            {
                annotation.metadata.get("type")
                for annotation in self.annotations
                if "type" in annotation.metadata  # Corrected this line
            }
        )

    def get_texts_of_annotation_type(self, annotation_set_type):
        """
        Retrieves the actual text of annotations of a specific annotation_set type.

        Args:
            annotation_set_type (str): The type of the annotation_set to retrieve.

        Returns:
            list: List of strings that are the actual extracted text of the annotation_set.
        """
        return [
            self.full_text[annotation.start : annotation.end + 1]
            for annotation in self.annotations
            if annotation.metadata.get("type")
            == annotation_set_type  # Corrected this line
        ]

    def get_texts_with_filter(
        self, predicate: Callable[[Dict[str, Any]], bool]
    ) -> List[str]:
        """
        Retrieves the actual text of annotations for which the predicate function returns True.

        Args:
            predicate (Callable[[Dict[str, Any]], bool]): A function that takes annotation metadata as input and returns a boolean.

        Returns:
            list: List of strings that are the actual extracted text of the annotations for which the predicate function returns True.
        """
        return [
            self.full_text[annotation.start : annotation.end]
            for annotation in self.annotations
            if predicate(annotation.metadata)
        ]

    def filter_by_metadata(self, key, value):
        """
        Filters annotations by metadata key-value pair.

        Args:
            key (str): The metadata key.
            value (str): The metadata value.

        Returns:
            list: List of annotations matching the key-value pair.
        """
        return [
            annotation
            for annotation in self.annotations
            if annotation.metadata.get(key) == value
        ]
