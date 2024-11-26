
import pickle
import numpy as np

from flask import Flask, request, jsonify


app = Flask('kickstart')

with open('model.pkl', 'rb') as f_in:
    model = pickle.load(f_in)


with open("vectorizer.pkl", "rb") as f_in:
    dv = pickle.load(f_in)


def predict_single(record, dv, model):
  X = dv.transform([record]) 
  y_pred = model.predict_proba(X)[:, 1]
  name = model.predict(X)

  return name,y_pred[0]


@app.route('/predict', methods=['POST'])  
def predict():
    record = request.get_json() 

    name , prediction = predict_single(record, dv, model)

    result = {
        'prediction_probability': float(prediction), 
        'prediction': str(name[0]),  
    }

    return jsonify(result)  ## send back the data in json format to the user


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)