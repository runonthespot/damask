from damask.models import Damask, Annotation
from damask.segmenters import SentenceSegmenter, WordSegmenter, ChunkSegmenter
from damask.annotators import PosAnnotator, SentimentAnnotator, LengthAnnotator
from damask.processors import AnnotationProcessor
import copy

damask = Damask(
    "This is a sentence in Belvedere. This is another sentence. This is a third sentence."
)

# create instances of segmenters
split_into_sentences = SentenceSegmenter()
split_into_words = WordSegmenter()
split_into_chunks = ChunkSegmenter(256)

damask.segment_text(segmenter=split_into_sentences)
damask.segment_text(segmenter=split_into_words)
damask.segment_text(segmenter=split_into_chunks)

# Create instances of your annotators
pos_annotator = PosAnnotator()
sentiment_annotator = SentimentAnnotator()
length_annotator = LengthAnnotator()

# Use the annotators to enrich the annotations
damask.enrich_annotations(enricher=pos_annotator, annotation_type="words")
damask.enrich_annotations(
    enricher=sentiment_annotator, annotation_type="sentences", parallel=True, workers=20
)

damask.enrich_annotations(enricher=length_annotator, annotation_type="chunks")

print("Displaying annotation sets as table:")
print(damask.annotation_sets_as_table())
print("done")

# print each chunk
print("Displaying chunk texts:")
for chunk in damask.chunks.texts:
    print(chunk)


# load PaulGrahamEssay.txt into damask

text = open("PaulGrahamEssay.txt", "r")
text = text.read()

essay = Damask(text)

essay.segment_text(segmenter=SentenceSegmenter())
essay.segment_text(segmenter=ChunkSegmenter(1024))


essay.enrich_annotations(
    enricher=sentiment_annotator, annotation_type="sentences", parallel=True, workers=20
)


chunks = essay.chunks.annotations
sentences = essay.sentences.annotations

"""
merged_chunks = essay.process_annotations_generic(
    chunks, sentences, merge_overlapping_sentences
)
"""

# write annotation_sets to a file
file = open("essay.txt", "w")
file.write(essay.annotation_sets_as_table())
file.close()
