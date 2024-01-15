from .base import Annotator
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import pos_tag
from nltk.data import load
from typing import List, Dict, Any, Callable


class PosAnnotator(Annotator):
    _resources_downloaded = False
    _tagdict = None
    _pos_tagger = None

    @classmethod
    def _ensure_resources_downloaded(cls):
        if not cls._resources_downloaded:
            nltk.download("averaged_perceptron_tagger")
            nltk.download("tagsets")
            cls._tagdict = load("help/tagsets/upenn_tagset.pickle")
            cls._pos_tagger = nltk.PerceptronTagger()
            cls._resources_downloaded = True

    def __init__(self):
        self._ensure_resources_downloaded()

    def annotate(self, text: str) -> Dict[str, Any]:
        # Use the tag method of the PerceptronTagger instance
        pos_tag_result = self._pos_tagger.tag([text])
        tag = pos_tag_result[0][1]
        tag_description = self._tagdict.get(tag, ("", ""))[0]
        return {"pos": tag, "pos_description": tag_description}


class SentimentAnnotator(Annotator):
    _resources_downloaded = False
    _sia = None

    @classmethod
    def _ensure_resources_downloaded(cls):
        if not cls._resources_downloaded:
            nltk.download("vader_lexicon")
            cls._sia = SentimentIntensityAnalyzer()
            cls._resources_downloaded = True

    def __init__(self):
        self._ensure_resources_downloaded()

    def annotate(self, text: str) -> Dict[str, Any]:
        sentiment_scores = self._sia.polarity_scores(text)
        return {"sentiment": sentiment_scores}
