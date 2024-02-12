from fastapi import FastAPI, UploadFile, File
from damask.models import Damask
from damask.segmenters import SentenceSegmenter, WordSegmenter, ChunkSegmenter
from damask.annotators import SentimentAnnotator, PosAnnotator
from typing import List

app = FastAPI()

# Store the Damask object in memory for this example
# In a production application, you would want to store this in a database
damask = None


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    global damask
    content = await file.read()
    text = content.decode("utf-8")
    damask = Damask(text)
    return {"status": "File uploaded"}


@app.get("/segment/")
def segment_text():
    global damask
    # Segment text
    split_into_sentences = SentenceSegmenter()
    split_into_words = WordSegmenter()
    split_into_chunks = ChunkSegmenter(1024)

    damask.segment_text(segmenter=split_into_sentences, annotation_type="sentences")
    damask.segment_text(segmenter=split_into_words, annotation_type="words")
    damask.segment_text(segmenter=split_into_chunks, annotation_type="chunks")
    return {"status": "Text segmented"}


@app.get("/annotate/")
def annotate_text():
    global damask
    # Annotate text
    damask.enrich_annotations(
        enricher=SentimentAnnotator(),
        annotation_type="sentences",
        parallel=True,
        workers=20,
    )
    damask.enrich_annotations(
        enricher=PosAnnotator(), annotation_type="words", parallel=True, workers=20
    )
    return {"status": "Text annotated"}


@app.get("/annotations/")
def get_annotations():
    global damask
    # Return annotated text
    return {
        "sentences": [
            annotation.to_dict() for annotation in damask.sentences.annotations
        ],
        "words": [annotation.to_dict() for annotation in damask.words.annotations],
        "chunks": [annotation.to_dict() for annotation in damask.chunks.annotations],
    }
