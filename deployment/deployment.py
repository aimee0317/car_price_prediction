# Author: Amelia Tang

from flask import Flask,request, url_for, redirect, render_template, jsonify
import pandas as pd
import pickle
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_transformer
from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# Initalize the Flask app
app = Flask(__name__, template_folder='templates')

# Loads pre-trained model
model = pickle.load(open('final_model.pkl', 'rb'))
cols = ["brand", "model", "year", "transmission", "mileage", "fuelType", "tax", "mpg", "engineSize"]

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = model.predict(data_unseen)
    prediction = int(prediction)
    return render_template('home.html',pred='Predicted used car price is {}'.format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = model.predict(data_unseen)
    output = prediction.Label[0]
    return jsonify(output)


if __name__ == '__main__':
    app.run(debug=True)
