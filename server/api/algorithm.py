import pickle
from keras.preprocessing.sequence import pad_sequences
from .helpers import clean_review

MAX_WORDS = 10000

class Algorithm:
    def __init__(self):
        self.tokenizer = None
        self.model = None

        self.load_tokenizer()
        self.load_model()
    
    def load_tokenizer(self):
        with open('tokenizer.pickle', 'rb') as file:
            self.tokenizer = pickle.load(file)

    def load_model(self):
        with open('model.pickle', 'rb') as file:
            self.model = pickle.load(file)

    def predict(self, review):
        print(review)
        review = clean_review(review)

        review_vec = self.tokenizer.texts_to_sequences([review])
        review_vec_pad = pad_sequences(review_vec, MAX_WORDS, padding='post')

        rating = self.model.predict(review_vec_pad)

        return int(rating[0])