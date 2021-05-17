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
en_nlp = spacy.load("en_core_web_sm")


class TfidfModel(BaseModel):
    def __init__(self,
                 clf,
                 config: PreprocessingConfig,
                 text_column_name: str,
                 max_df=0.3,
                 max_features=5000
                 ):
        super().__init__(config, [text_column_name])
        self.clf = clf
        self.vectorizer = TfidfVectorizer(max_features=max_features,
                                          max_df=max_df,
                                          analyzer='word')

    def _replace_regexps(self, doc):
        doc = re.sub(EXCEPTION_REGEXP, " EXCEPTION_STACK_TRACE ", doc)
        doc = re.sub(LINK_REGEXP, " URL_LINK ", doc)
        doc = re.sub(MARKDOWN_LINK_REGEXP, " MARKDOWN_LINK ", doc)
        doc = re.sub(MARKDOWN_PHOTO_REGEXP, " MARKDOWN_PHOTO ", doc)
        return doc

    def _preprocess(self, data: pd.Series) -> pd.Series:
        data = data.apply(self._replace_regexps)
        docs = en_nlp.pipe(data)
        data_list = [
            ' '.join([ent.lemma_.lower() for ent in doc if ent.lemma_.isalpha()]) + ' '
            for doc in docs
        ]
        return pd.Series(data_list, index=data.index)

    def _fit(self, full_data: Dict[str, pd.DataFrame]) -> None:
        data, target = self.to_data_target(full_data['train'])
        data = self.unite_pairs(data)
        data = data[self.features[0]]
        data_transformed = self.vectorizer.fit_transform(data)
        data_transformed = self.concat_pairs(data_transformed.toarray())
        self.clf.fit(data_transformed, target)

    def _predict_proba(self, data: pd.DataFrame) -> np.array:
        data = self.unite_pairs(data)
        data = data[self.features[0]]
        data_transformed = self.vectorizer.transform(data)
        data_transformed = self.concat_pairs(data_transformed.toarray())
        return self.clf.predict_proba(data_transformed)

