# Author: Amelia Tang

"""Conduct hyperparameter tunining on the best model and save the final model in the pickle format. 

Usage: src/final_model.py --model_path=<model_path>

Options:
--model_path=<model_path>   path to save the final model in the pickle format
"""


import os
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import (
    cross_val_score,
    cross_validate,
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor
import pickle
from docopt import docopt

opt = docopt(__doc__)


def main(model_path):
    # read train data
    X_train = pd.read_csv("data/raw/X_train.csv", parse_dates=['year'])
    X_train['year'] = X_train['year'].dt.year
    y_train = pd.read_csv("data/raw/y_train.csv")
    train_df = X_train.join(y_train.set_index('carID'), on="carID")

    # separate X and y
    X_train, y_train = train_df.drop(columns="price"), train_df["price"]

    # hyperparameter optimization and save the final model in the pickle format
    final_model_pickle = final_model(train_df, X_train, y_train)
    pickle.dump(final_model_pickle, open(model_path, "wb"))


def final_model(train_df, X_train, y_train):
    """
    Conduct hyperparameter optimization on the best model and return the tuned final model 

    Parameters
    ----------
    train_df : dataframe
        training dataset 
    X_train: dataframe 
        Predicting variables in the training dataset 
    y_train : dataframe
        Targets in the training dataset 
    Returns
    -------
    final_model 
        the final model after hyperparameter tuning
    """

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
        (OrdinalEncoder(handle_unknown="use_encoded_value",
         unknown_value=np.nan), ordinal_features),
        ("drop", drop_features))

    params = {'xgbregressor__max_depth': np.arange(1, 20, 1),
              'xgbregressor__learning_rate': np.linspace(0.01, 1, 20),
              'xgbregressor__colsample_bytree': [0.3, 0.7]}

    pipe_xgboost = make_pipeline(preprocessor, XGBRegressor(random_state=123))

    best_model = RandomizedSearchCV(
        pipe_xgboost,
        params,
        cv=3,
        scoring="neg_root_mean_squared_error",
        return_train_score=True,
        random_state=123)

    final_model = best_model.fit(X_train, y_train)
    # print(best_model.best_score_)

    return final_model


if __name__ == "__main__":
    main(opt["--model_path"])
