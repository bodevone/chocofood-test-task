import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def clean_review(review):
    review = review.lower()
    review = word_tokenize(review)
    return review