from flask import Flask, request, render_template
import numpy as np
import pickle
import tensorflow

app = Flask(__name__, static_folder='static')

import re
import tensorflow
import pickle
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean_text(text):
    # Remove URLs, mentions, special characters and convert to lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text


def gettextprocessed(tweet):
    file = open("tokenizer.pickle", 'rb')
    tokenizer = pickle.load(file)
    text = pd.Series([tweet])
    text = text.apply(clean_text)
    temp = tokenizer.texts_to_sequences(text)
    temp_padded = pad_sequences(temp, maxlen=50, padding='post', truncating='post')
    return temp_padded


# Load the trained model
model = tensorflow.keras.models.load_model("Final_Model.h5")


# Define the function for predicting sentiment
def predict_sentiment(tweet):
    predicted = model.predict(gettextprocessed(tweet))
    return predicted[0]


# Define the route for the homepage
@app.route('/')
def home():
    return render_template('home.html')


# Define the route for the prediction
@app.route('/predict', methods=['POST'])
def predict():
    tweet = request.form['tweet']
    sentiment = predict_sentiment(tweet)
    return render_template('result.html', tweet=tweet, sentiment=sentiment)


if __name__ == '__main__':
    app.run(debug=True)
