from typing import List, Dict, Any, Callable, Tuple, Optional
from damask.models.annotation import Annotation


class AnnotationSetView:
    """
    Represents a view of a subset of annotations from a Damask instance.

    This class facilitates access to annotations of a specific type within a Damask instance.
    It provides
    -methods to retrieve both the annotations and the corresponding text segments.
    -ability to transform the annotation bounds using a provided lambda function.

    Attributes:
        damask_instance (Damask): The Damask instance containing the annotations.
        annotation_type (str): The type of annotations to be viewed.
        transform (Callable[[int, int], Tuple[int, int]], optional): A function to transform
            the annotation bounds. It takes start and end indices and returns a tuple of
            transformed indices.

    Methods:
        annotations: Returns a list of annotations of the specified type, optionally
            transformed using the provided transform function.
        texts: Returns a list of text segments corresponding to the annotations.
        __iter__: Returns an iterator over the annotations.
        __len__: Returns the number of text segments corresponding to the annotations.
    """

    def __init__(
        self,
        damask_instance: "Damask",
        annotation_type: str,
        transform: Callable[[int, int], Tuple[int, int]] = None,
    ):
        self.damask_instance = damask_instance
        self.annotation_type = annotation_type
        self.transform = transform

    @property
    def annotations(self) -> List[Annotation]:
        annotations = [
            annotation
            for annotation in self.damask_instance.annotations
            if annotation.metadata.get("type") == self.annotation_type
        ]
        if self.transform:
            return [
                annotation.adjust_bounds(
                    self.transform, len(self.damask_instance.full_text)
                )
                for annotation in annotations
            ]
        return annotations

    @property
    def texts(self) -> List[str]:
        return [
            self.damask_instance.full_text[annotation.start : annotation.end]
            for annotation in self.annotations
        ]

    def apply(
        self, transform: Callable[[int, int], Tuple[int, int]]
    ) -> "AnnotationSetView":
        # Apply the transformation and return a new AnnotationSetView
        transformed_annotations = [
            Annotation(transform(annotation.start, annotation.end), annotation.metadata)
            for annotation in self.annotations
        ]
        # Create a new Damask instance with the transformed annotations
        new_damask = Damask(self.damask_instance.full_text)
        new_damask.annotations = transformed_annotations
        return AnnotationSetView(new_damask, self.annotation_type)

    def __iter__(self):
        return iter(self.annotations)

    def __len__(self):
        return len(self.texts)
