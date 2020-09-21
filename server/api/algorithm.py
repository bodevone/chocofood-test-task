import os
import pickle

from keras.preprocessing.sequence import pad_sequences
from django.conf import settings

from api.helpers import clean_review


class Algorithm:
    MAX_WORDS = 10000
    TOKENIZER_FILE_NAME = 'tokenizer.pickle'
    MODEL_FILE_NAME = 'model.pickle'
    TOKENIZER_PATH = os.path.join(settings.ALGORITHM_ROOT, TOKENIZER_FILE_NAME)
    MODEL_PATH = os.path.join(settings.ALGORITHM_ROOT, MODEL_FILE_NAME)

    def __init__(self):
        self.tokenizer = None
        self.model = None

        self.load_tokenizer()
        self.load_model()
    
    def load_tokenizer(self):
        with open(self.TOKENIZER_PATH, 'rb') as f:
            self.tokenizer = pickle.load(f)

    def load_model(self):
        with open(self.MODEL_PATH, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, review):
        review = clean_review(review)

        review_vec = self.tokenizer.texts_to_sequences([review])
        review_vec_pad = pad_sequences(review_vec, self.MAX_WORDS, padding='post')

        rating = self.model.predict(review_vec_pad)

        return int(rating[0])