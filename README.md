# Used Car Price Prediction

-   Author: Amelia Tang 

## About

I built a regression model to predict the price of used cars using features relevant to evaluate the quality of used cars, such as the brand, model, year built, the size of the engine and the type of fuel used. I experimented on four popular algorithms for regression problems: linear regression with L2 regularization (ridge), linear regression with L1 regularization (lasso), random forest and CatBoost. I also utilized scikit-learn's `DummyRegressor` as a base case for comparison. After comparing train and test $R^2$, I selected CatBoost as the best algorithm to use. The CatBoost model performed well with an $R^2$ as 96.8%. 

The data set used in this project was a subset of the `100,00 UK Used Car Data set` on kaggle.com and available [here](https://www.kaggle.com/kukuroo3/used-car-price-dataset-competition-format). Each row of the data represents a used car and provides its ID, brand, model, year, transmission, mileage, fuel type, tax, miles per gallon and engine size. 
