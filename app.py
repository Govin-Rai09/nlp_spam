from flask import Flask, redirect, render_template, request, url_for
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

import pickle
tfid = pickle.load(open('model.pkl', 'rb'))
clf = pickle.load(open('nlp.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        data = [text]
        vect = tfid.transform(data).toarray()
        pred = clf.predict(vect)
    return render_template('result.html', prediction=pred)


if __name__ == '__main__':
    app.run(debug=True)
