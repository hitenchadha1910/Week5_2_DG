import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def home():
    if(request.method == "GET"):

        data = "Hello"
        return jsonify({'data': data})
    

@app.route('/predict/')
def predict():

    model = pickle.load(open('model.pkl', 'rb'))
    experience = request.args.get('Experience')
    test_score = request.args.get('Test Score')

    df = pd.DataFrame({'experience':[experience], 'test_score':[test_score]})

    prediction = model.predict(df)

    return jsonify({'Salary predicted': str(prediction)})

if __name__ == '__main__':
    app.run(debug=True)