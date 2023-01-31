# Author: Amelia Tang

"""Evaluate the final model on the test data 

Usage: src/test_result.py --test_x=<test_x> --test_y=<test_y> --model_path=<model_path> --csv_path=<csv_path>

Options:
--test_x=<test_x>   Path to the test data x
--test_y=<test_y>   Path to the test data y
--model_path=<model_path> Path to the model in the pickle format 
--csv_path=<csv_path> Path to save the test scores csv file  
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import pickle
from docopt import docopt

opt = docopt(__doc__)


def main(test_x, test_y, model_path, csv_path):
    # read test data
    X_test = pd.read_csv(test_x, parse_dates=['year'])
    X_test['year'] = X_test['year'].dt.year
    y_test = pd.read_csv(test_y)['price']

    # Load model and predict
    final_model = pickle.load(open(model_path, 'rb'))
    y_predict = final_model.predict(X_test)

    # Calculate the test scores
    Neg_RMSE = - np.sqrt(mean_squared_error(y_test, y_predict))
    r2 = r2_score(y_test, y_predict)

    # Make test score data frame
    scores_df = pd.DataFrame(
        {'Neg_RMSE': [Neg_RMSE],
         'R_squared': [r2]}
    )

    # Save test scores to csv
    try:
        scores_df.to_csv(csv_path)
    except:
        os.makedirs(os.path.dirname(csv_path))
        scores_df.to_csv(csv_path)


if __name__ == "__main__":
    main(opt["--test_x"], opt["--test_y"],
         opt["--model_path"], opt["--csv_path"])
