from abc import ABC, abstractmethod
from typing import Dict, Any
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import pos_tag
from nltk.data import load
from damask.models.annotator import Annotator


class PosAnnotator(Annotator):
    def annotate(self, text: str) -> Dict[str, Any]:
        pos_tag_result = pos_tag([text])
        tag = pos_tag_result[0][1]
        tag_description = tagdict.get(tag, ("", ""))[0]
        return {"pos": tag, "pos_description": tag_description}


class SentimentAnnotator(Annotator):
    def annotate(self, text: str) -> Dict[str, Any]:
        sia = SentimentIntensityAnalyzer()
        sentiment_scores = sia.polarity_scores(text)
        return {"sentiment": sentiment_scores}


class LengthAnnotator(Annotator):
    def annotate(self, text: str) -> Dict[str, Any]:
        return {"length": len(text)}
