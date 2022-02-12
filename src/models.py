# Author: Amelia Tang

"""Trains the models and saves the training and validation scores to a csv file. Saves the final model's shap plot as png.
Saves the test scores of the tuned model to a csv file.

Usage: models_selection.py --csv_path=<csv_path> --shap_path=<shap_path> --score_path=<score_path>

Options:
--csv_path=<csv_path>   path and file name of the model scores csv file
--shap_path=<shap_path> path and filename of the shap plot of the tuned model 
--score_path=<score_path> path and filename of the test scores csv file
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


def main(csv_path, shap_path, score_path):
    # read train data
    X_train = pd.read_csv("data/raw/X_train.csv", parse_dates=['year'])
    X_train['year'] = X_train['year'].dt.year
    y_train = pd.read_csv("data/raw/y_train.csv")
    train_df = X_train.join(y_train.set_index('carID'), on = "carID")
    
    # read test data 
    X_test = pd.read_csv("data/raw/X_test.csv", parse_dates=['year'])
    X_test['year'] = X_test['year'].dt.year
    y_test = pd.read_csv("data/raw/y_test.csv")
  
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
        
    # test score
    test_score = tuned_model(preprocessor, train_df, X_train, y_train, X_test, y_test, shap_path)
    try:
        test_score.to_csv(score_path, encoding="utf-8")
    except:
        os.makedirs(os.path.dirname(score_path))
        open(score_path, "wb").write(test_score.content)


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


def tuned_model(preprocessor, train_df, X_train, y_train, X_test, y_test, shap_path):
  preprocessor.fit(X_train)
  pipe_lr = make_pipeline(
    preprocessor, Ridge(random_state=123)
  )
  
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
  
  feature_names = (
    numeric_features
    + list(
        pipe_lr.named_steps["columntransformer"]
        .named_transformers_["onehotencoder"]
        .get_feature_names_out()
    )
    + ordinal_features)
    
  X_train_encode = pd.DataFrame(
    data=preprocessor.transform(X_train).toarray(),
    columns=feature_names,
    index=X_train.index
    )
  
  tuned_pipe_catboost = make_pipeline(
    preprocessor, CatBoostRegressor(random_state=123, max_depth = 8, verbose = 0)
  )
  
  # Generate shap plot and save the plot 
  tuned_pipe_catboost.fit(X_train, y_train)
  catboost_explainer = shap.TreeExplainer(tuned_pipe_catboost.named_steps["catboostregressor"])
  train_catboost_shap_values = catboost_explainer.shap_values(X_train_encode)
  
  shap.summary_plot(train_catboost_shap_values, X_train_encode, show=False)
  plt.savefig(shap_path, bbox_inches='tight')
  
  
  y_predict = tuned_pipe_catboost.predict(X_test)
  R_squared = r2_score(y_test['price'], y_predict)
  Root_mean_squared_error = np.sqrt(mean_squared_error(y_test['price'], y_predict))
  data = [[R_squared, Root_mean_squared_error]]
  return pd.DataFrame(data, columns = ['R_squared', 'Root mean squared error'])
  

if __name__ == "__main__":
    main(opt["--csv_path"], opt["--shap_path"], opt["--score_path"])
