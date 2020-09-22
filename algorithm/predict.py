import pickle
from keras.preprocessing.sequence import pad_sequences
from helpers import clean_review
from model import MAX_WORDS

with open('data/tokenizer.pickle', 'rb') as file:
    tokenizer = pickle.load(file)

with open('data/model.pickle', 'rb') as file:
    model = pickle.load(file)

def predict(review):
    review = clean_review(review)

    review_vec = tokenizer.texts_to_sequences([text])
    review_vec_pad = pad_sequences(review_vec, MAX_WORDS, padding='post')

    rating = model.predict(review_vec_pad)

    return int(rating[0])

if __name__ == '__main__':
    text = 'Было больше 40-ка минут. Кола тёплая, фри остыло.'
    rating = predict(text)
    print(f'Predicted rating: {rating}')
