import logging
import pickle
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from helpers import clean_review

MAX_WORDS = 10000

def model():
    logging.info('Reading Data')
    data = pd.read_csv('data/food-review202009141259.csv', skipinitialspace=True)
    reviews = data['text']
    ratings = data['rating']

    logging.info('Cleaning reviews')
    reviews = reviews.apply(clean_review)

    logging.info('Tokenizing reviews')
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(reviews)

    logging.info('Saving tokenizer')
    with open('data/tokenizer.pickle', 'wb') as file:
        pickle.dump(tokenizer, file, protocol=pickle.HIGHEST_PROTOCOL)

    reviews_vec = tokenizer.texts_to_sequences(reviews)
    reviews_vec_pad = pad_sequences(reviews_vec, MAX_WORDS, padding='post')

    logging.info('Splitting data')
    X_train, X_test, y_train, y_test = train_test_split(
        reviews_vec_pad,
        ratings,
        test_size=0.1, 
        random_state=42
    )

    logging.info('Building model')
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    print(mean_squared_error(y_test, y_pred_test))

    logging.info('Saving model')
    with open('data/model.pickle', 'wb') as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    model()