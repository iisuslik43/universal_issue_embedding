import re

import numpy as np
import pandas as pd
from typing import List, Dict

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.tokens import Doc
import spacy
from scripts.dataset.preprocessing.config import PreprocessingConfig
from scripts.models.base_model import BaseModel

EXCEPTION_REGEXP = r"[\w\.]+Exception[^\n]*\n(\s+at .+)+"
LINK_REGEXP = r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
MARKDOWN_LINK_REGEXP = r"\[(.*?)\]\(.*?\)"
MARKDOWN_PHOTO_REGEXP = r"!\[(.*?)\]\(.*?\)"


try:
    en_stops = set(stopwords.words('english'))
except LookupError:
    import nltk
    nltk.download('stopwords')
    en_stops = set(stopwords.words('english'))
en_nlp = spacy.load('en')


class TfidfModel(BaseModel):
    def __init__(self,
                 clf,
                 config: PreprocessingConfig,
                 text_column_name: str,
                 fit_params=None,
                 max_df=0.3,
                 max_features=2000
                 ):
        super().__init__(config, [text_column_name])
        if fit_params is None:
            fit_params = {}
        self.clf = clf
        self.fit_params = fit_params
        self.vectorizer = TfidfVectorizer(max_features=max_features,
                                          max_df=max_df,
                                          analyzer='word',
                                          tokenizer=self._tokenize,
                                          preprocessor=self._preprocess)

    def _preprocess(self, doc):
        doc = re.sub(EXCEPTION_REGEXP, " EXCEPTION_STACK_TRACE ", doc)
        doc = re.sub(LINK_REGEXP, " URL_LINK ", doc)
        doc = re.sub(MARKDOWN_LINK_REGEXP, " MARKDOWN_LINK ", doc)
        doc = re.sub(MARKDOWN_PHOTO_REGEXP, " MARKDOWN_PHOTO ", doc)
        return doc

    def _tokenize(self, doc):
        words = re.split(r'[^\w]+', doc)
        words = [word.lower() for word in words if len(word) >= 3]
        words = [token.lemma_ for token in Doc(en_nlp.vocab, words=words)]
        return [word for word in words if word not in en_stops]

    def fit(self, data: Dict[str, pd.DataFrame]) -> None:
        data, target = self.to_data_target(data['train'])

    def transform(self, data: pd.DataFrame) -> np.array:
        data, _ = self.to_data_target(data['train'])
