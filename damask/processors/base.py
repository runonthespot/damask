from typing import List
from damask.models.annotation import Annotation
from typing import Dict, Any

"""
Annotation Properties, Methods and Attributes
=============================================
Properties:
    start (int): The start index of the annotation.
    end (int): The end index of the annotation.
    metadata (dict): The metadata of the annotation.
    length (int): The length of the annotation.
Methods:
    overlaps_with(other: Annotation) -> bool: Returns True if the annotation overlaps with the other annotation.
"""


class AnnotationProcessor:
    def __init__(self, annotations: List[Annotation]):
        self.annotations = annotations

    @staticmethod
    def overlaps(ann1: Annotation, ann2: Annotation) -> bool:
        return ann1.overlaps_with(
            ann2
        )  # Use the overlaps_with method from the Annotation class

    @staticmethod
    def merge_metadata(
        existing_metadata: Dict[str, Any], addition_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Create a copy of the existing metadata to avoid modifying the original
        merged_metadata = existing_metadata.copy()
        # Get the type from the addition's metadata
        addition_type = addition_metadata.get("type")
        if addition_type:
            # If the type key doesn't exist in the existing metadata, initialize it as an empty list
            if addition_type not in merged_metadata:
                merged_metadata[addition_type] = []
            # Append the existing metadata to the list under the new key
            merged_metadata[addition_type].append(existing_metadata)
        return merged_metadata

    @staticmethod
    def merge_into(existing: Annotation, addition: Annotation) -> List[Annotation]:
        if AnnotationProcessor.overlaps(existing, addition):
            # Merge the metadata using the new merge_metadata function
            merged_metadata = AnnotationProcessor.merge_metadata(
                existing.metadata, addition.metadata
            )
            # Use the merge_with method from the Annotation class, passing the merged metadata
            return [
                Annotation(
                    start=min(existing.start, addition.start),
                    end=max(existing.end, addition.end),
                    metadata=merged_metadata,
                )
            ]
        return []

    @staticmethod
    def subtract(
        target_annotation: Annotation, subtracting_annotation: Annotation
    ) -> List[Annotation]:
        if not AnnotationProcessor.overlaps(target_annotation, subtracting_annotation):
            # If they don't overlap, return the target annotation as is
            return [target_annotation]

        new_annotations = []
        # Case where subtracting_annotation is completely inside target_annotation, resulting in two new annotations
        if (
            subtracting_annotation.start > target_annotation.start
            and subtracting_annotation.end < target_annotation.end
        ):
            new_annotations.append(
                Annotation(
                    start=target_annotation.start,
                    end=subtracting_annotation.start,
                    metadata=target_annotation.metadata.copy(),
                )
            )
            new_annotations.append(
                Annotation(
                    start=subtracting_annotation.end,
                    end=target_annotation.end,
                    metadata=target_annotation.metadata.copy(),
                )
            )
        # Case where subtracting_annotation overlaps the start of target_annotation
        elif (
            subtracting_annotation.start <= target_annotation.start
            and subtracting_annotation.end < target_annotation.end
        ):
            new_annotations.append(
                Annotation(
                    start=subtracting_annotation.end,
                    end=target_annotation.end,
                    metadata=target_annotation.metadata.copy(),
                )
            )
        # Case where subtracting_annotation overlaps the end of target_annotation
        elif (
            subtracting_annotation.start > target_annotation.start
            and subtracting_annotation.end >= target_annotation.end
        ):
            new_annotations.append(
                Annotation(
                    start=target_annotation.start,
                    end=subtracting_annotation.start,
                    metadata=target_annotation.metadata.copy(),
                )
            )
        # Case where subtracting_annotation covers target_annotation completely, resulting in no annotations
        elif (
            subtracting_annotation.start <= target_annotation.start
            and subtracting_annotation.end >= target_annotation.end
        ):
            return []

        return new_annotations
