# Author: Amelia Tang

"""Reads train_data csv file, perform scoring on multiple classification models, 
and then select the results in a csv file. Then, it will run hyperprameter optimization
on the best model and saves the model in a pickle format

Usage: models_selection.py --csv_path=<csv_path>

Options:
--csv_path=<csv_path>   path and file name of the model_selection scores csv file
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import altair as alt
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import (
 cross_val_score,
 cross_validate,
 GridSearchCV
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import shap

from docopt import docopt

opt = docopt(__doc__)


def main(csv_path):
    # read train data
    X_train = pd.read_csv("data/raw/X_train.csv", parse_dates=['year'])
    X_train['year'] = X_train['year'].dt.year
    y_train = pd.read_csv("data/raw/y_train.csv")
    train_df = X_train.join(y_train.set_index('carID'), on = "carID")
  
    # separate X and y
    X_train, y_train = train_df.drop(columns="price"), train_df["price"]

    # define preprocessor
    preprocessor = preprocess(train_df)

    # generate model scores
    results = generate_scores(preprocessor, X_train, y_train)
    try:
        results.to_csv(csv_path, encoding="utf-8")
    except:
        os.makedirs(os.path.dirname(csv_path))
        open(csv_path, "wb").write(results.content)


def preprocess(train_df):
    # split columns into different categories 
    drop_features = ["carID"]
    categorical_features = ["brand", "model", "transmission", "fuelType"]
    ordinal_features = ["year"]
    target = "price"
    numeric_features = list(
      set(train_df.columns)
      - set(drop_features)
      - set(categorical_features)
      - set(ordinal_features)
      - set([target])
    )
    
    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_features),
        (OneHotEncoder(handle_unknown="ignore", dtype="int"), categorical_features),
        (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), ordinal_features),
        ("drop", drop_features))

    return preprocessor
  

def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
    """
    Returns mean and std of cross validation

    Parameters
    ----------
    model :
        scikit-learn model
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train :
        y in the training data

    Returns
    ----------
        pandas Series with mean scores from cross_validation
    """

    scores = cross_validate(model, X_train, y_train, **kwargs)

    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores[i], std_scores[i])))

    return pd.Series(data=out_col, index=mean_scores.index)


def generate_scores(preprocessor, X_train, y_train):
    scoring_metrics = {
    "neg RMSE": "neg_root_mean_squared_error",
    "r2": "r2"}

    # base line
    results ={}
    dummy = DummyRegressor()
    results["Dummy"] = mean_std_cross_val_scores(
      dummy, X_train, y_train, return_train_score=True, scoring=scoring_metrics
    )
    
    # model pipelines
    pipe_lr = make_pipeline(
        preprocessor, Ridge(random_state=123)
    )

    pipe_ls = make_pipeline(
        preprocessor, Lasso(random_state=123)
    )

    pipe_rf = make_pipeline(
        preprocessor, RandomForestRegressor(random_state=123)
    )

    pipe_catboost = make_pipeline(
        preprocessor, CatBoostRegressor(random_state=123, verbose = 0)
    )
    
    # models 
    models = {
        "ridge": pipe_lr,
        "lasso": pipe_ls,
        "random forest": pipe_rf,
        "CatBoost": pipe_catboost
    }

    # generate results
    for (name, model) in models.items():
      results[name] = mean_std_cross_val_scores(
        model, X_train, y_train, return_train_score=True, scoring=scoring_metrics
      )

    return pd.DataFrame(results)


if __name__ == "__main__":
    main(opt["--csv_path"])
