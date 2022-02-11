# Used Car Price Prediction

-   Author: Amelia Tang 

## About

I built a regression model to predict the price of used cars using features relevant to evaluate the quality of used cars, such as the brand, model, year built, the size of the engine and the type of fuel used. I experimented on four popular algorithms for regression problems: linear regression with L2 regularization (`ridge`), linear regression with L1 regularization (`lasso`), `random forest` and `CatBoost`. I also utilized scikit-learn's `DummyRegressor` as a base case for comparison. After comparing training and test R-squared scores, I selected CatBoost as the best algorithm to use. The CatBoost model performed well with an R-squared of 96.8%. 

The data set used in this project was a subset of the `100,00 UK Used Car Data set` on kaggle.com and available [here](https://www.kaggle.com/kukuroo3/used-car-price-dataset-competition-format). Each row of the data represents a used car and provides its ID, brand, model, year, transmission, mileage, fuel type, tax, miles per gallon and engine size. 

## Report

The final report can be found here.

## Usage

### Creating the environment

`conda env create --file car_env.yaml`

Run the following command from the environment where you installed
JupyterLab.

`conda install nb_conda_kernels`

If you are a windows user, run the following command inside the newly
created environment to install vega-lite.

`npm install -g vega vega-cli vega-lite canvas`

For M1 mac users, make sure you are using the `x86` version of conda and
not the `arm64` version. See
[here](https://github.com/conda-forge/miniforge#miniforge3) and
[here](https://github.com/mwidjaja1/DSOnMacARM/blob/main/README.md) for
more info.

### To replicate the analysis
Clone this Github repository, install the dependencies, and run the 
following commands at the command line/terminal from the root directory of the project:

    make all

To reset the repo to the original state, and delete all results files
and report, run the following commands at the command line/terminal from
the root directory of the project:

    make clean

## Dependencies

A complete list of dependencies is available [here](https://github.com/aimee0317/car_price_prediction/blob/main/car_env.yaml).
<br>- Python 3.9.7 and Python packages: <br>- docopt==0.6.1 <br>-
pandas==1.3.3 <br>- numpy==1.21.2 <br>- altair_saver=0.5.0 <br>-
altair=4.1.0 <br>- scikit-learn=1.0


## Replicating the analysis using Docker
If you would like to replicate the analysis using Docker, follow the steps below:

From the root of this project, run the following command to replicate the analysis:

`docker-compose run --rm report-env make -C //home//xxx//work all`

To reset the project to the original state, and delete all result files and report, 
run the following command:

`docker-compose run --rm report-env make -C //home//xxx//work clean`
