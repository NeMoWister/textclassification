import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

stop_words = set(stopwords.words('english'))

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, empty_token='[empty]'):
        self.empty_token = empty_token

    def _clean_one(self, text):
        text = "" if text is None else text
        text = str(text)
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = re.sub('\s+', ' ', text).strip() # тут удаляется всякий мусор
        if text == "":
            return self.empty_token
        words = [w for w in text.split() if w not in stop_words]
        if not words:
            return self.empty_token
        return " ".join(words)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, (pd.Series, pd.DataFrame)):
            X = X.values
        X = np.asarray(X, dtype=object)
        cleaned = [self._clean_one(x) for x in X]
        return np.array(cleaned, dtype=object)
    

class KerasTokenizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, max_length=None, num_words=None, oov_token='[OOV]', empty_token='[empty]'): # тут применяется обычный керас Tokenizer
        self.max_length = max_length # длинна последовательности
        self.num_words = num_words # размер словаря
        self.oov_token = oov_token
        self.empty_token = empty_token
        self.tokenizer = None
        self.vocab_size = None

    def fit(self, X, y=None):
        X_list = ["" if x is None else str(x) for x in X]
        self.tokenizer = Tokenizer(num_words=self.num_words, oov_token=self.oov_token) # ограничим размер словаря 10к
        self.tokenizer.fit_on_texts(X_list)

        if self.num_words is not None:
            self.vocab_size = min(self.num_words, len(self.tokenizer.word_index) + 1)
        else:
            self.vocab_size = len(self.tokenizer.word_index) + 1

        sequences = self.tokenizer.texts_to_sequences(X_list) # здесь все преобразуется в последовательности для обучения
        lengths = [len(s) for s in sequences]
        if self.max_length is None:
            self.max_length = int(max(1, max(lengths) if lengths else 1))
        else:
            self.max_length = int(self.max_length)
        return self

    def transform(self, X):
        X_list = ["" if x is None else str(x) for x in X]
        sequences = self.tokenizer.texts_to_sequences(X_list)
        padded = pad_sequences(
            sequences,
            maxlen=self.max_length,
            padding='post',
            truncating='post',
            dtype='int32'
        )
        return padded
    

CustomPipeline = Pipeline([
    ("preprocess", TextPreprocessor()),
    ("tokenizer", KerasTokenizerTransformer())
])
