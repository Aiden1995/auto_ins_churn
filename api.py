# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 17:07:59 2018

@author: Sourabh Jana
"""

# Dependencies
from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd


# Your API definition
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if classifier:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(classifier.predict(query))
            confidence = classifier.predict_proba(query)
            return jsonify({'prediction': str(prediction)},{'confidence': str(confidence)})
           
        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    classifier = joblib.load("classification_model.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')

    app.run(port=port, debug=True)