# Author: Amelia Tang

"""Trains the models and saves the training and validation scores to a csv file. Saves the final model's shap plot as png.
Saves the test scores of the tuned model to a csv file.

Usage: src/shap_plot.py --train_x=<train_x> --train_y=<train_y> --shap_path=<shap_path> 

Options:
--train_x=<train_x>   path and file name of the training data x portion (features)
--train_y=<train_y> path and filename of the training data y portion (target)
--shap_path=<shap_path> path and filename to save the shap plot 
"""

from docopt import docopt
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import shap
import warnings
warnings.filterwarnings('ignore')


opt = docopt(__doc__)


def main(train_x, train_y, shap_path):
    # read train data
    X_train = pd.read_csv(train_x, parse_dates=['year'])
    X_train['year'] = X_train['year'].dt.year
    y_train = pd.read_csv(train_y)
    train_df = X_train.join(y_train.set_index('carID'), on="carID")

    # separate X and y
    X_train, y_train = train_df.drop(columns="price"), train_df["price"]

    # shap plot
    shap_plot(train_df, X_train, y_train)
    plt.savefig(shap_path, bbox_inches='tight')


def shap_plot(train_df, X_train, y_train):
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

    pipe_xgb = make_pipeline(
        preprocessor, XGBRegressor(random_state=123)
    )

    # Generate shap plot and save the plot
    pipe_xgb.fit(X_train, y_train)
    xgboost_explainer = shap.TreeExplainer(
        pipe_xgb.named_steps["xgbregressor"])
    shap_values = xgboost_explainer.shap_values(X_train_encode)
    return shap.summary_plot(shap_values, X_train_encode, show=False)


if __name__ == "__main__":
    main(opt["--train_x"], opt["--train_y"], opt["--shap_path"])
