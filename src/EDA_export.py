"""
EDA plots
Usage: EDA_export.py --plot1_path=<plot1_path> --plot2_path=<plot2_path> --plot3_path=<plot3_path>
Options:
--plot1_path=<plot1_path>              file path of the target_distribution_plot
--plot2_path=<plot2_path>              file path of the price_by_brand
--plot3_path=<plot3_path>              file path of corr_plot
"""


import os
import pandas as pd 
import numpy as np 
import altair as alt
from docopt import docopt


opt = docopt(__doc__)

def main(plot1_path, plot2_path, plot3_path):
  """
  Create the plots and save them as png files

  
  Parameters
  -----
  plot1_path : str
        file path to save the plot
  plot2_path : str
        file path to save the plot
  plot3_path : str
        file path to save the plot
  plot1 = target_distribution_plot(data)
  plot2 = price_by_brand(data)
  plot3 = corr_plot(data)
  """
  X_train = pd.read_csv("data/raw/X_train.csv", parse_dates=['year'])
  X_train['year'] = X_train['year'].dt.year
  y_train = pd.read_csv("data/raw/y_train.csv")
  train_df = X_train.join(y_train.set_index('carID'), on = "carID")

  target_distribution_plot(data)
  price_by_brand(data)
  corr_plot(data)

# outputs 
  save_plots(plot1, plot1_path, 4)
  save_plots(plot2, plot2_path, 4)
  save_plots(plot3, plot3_path, 4)
  
# Altair Plots
def target_distribution_plot(data):
  """
  Generate a histagram for the distribution of the targert column 
  
  Parameters:
  --------
  data : dataframe
    the data used to generate the histagram 
    
  Returns:
  --------
  Altair plot
    A histagram of the values in the target column
   """
    
  plot1 = alt.Chart(train_df, title ="Used Car Price Distribution").mark_bar().encode(\
  alt.X('price', bin=alt.Bin(maxbins=30), title="Price"),\
  y='count()')
    
  return plot1
  
def price_by_brand(data):
  """
  Generate boxplots to show the distributions of price by brand 
  
  Parameters: 
  ------
  data : dataframe 
    the data used to generate the boxplots 
  
  Returns:
  -------
  Altair plot
    Side-by-side boxplots 
  """
  
  plot2 = alt.Chart(train_df, title ="Scatter plot of Price vs. Year").mark_point().encode(\
  alt.X('year',scale=alt.Scale(zero=False)),\
  y='price')
  
  return plot2 

def corr_plot(data):
  """
  Generate correlation plots among numeric variables 
  
  Parameters:
  -----
  data: dataframe
    the data used to generate the correlation plots 
    
  Returns:
  -----
  Altair plot 
    Correlation plots 
  """
  
  plot3 = alt.Chart(train_df).mark_point(opacity=0.3, size=10).encode(
     alt.X(alt.repeat('row'), type='quantitative', scale=alt.Scale(zero=False)),
     alt.Y(alt.repeat('column'), type='quantitative', scale=alt.Scale(zero=False))
    ).properties(
    width=200,
    height=200
    ).repeat(
    column=['price', 'year', 'mpg'],
    row=['price', 'year', 'mpg']
     )
     
  return plot3 

# Save plots
def save_plots(plot, file_path, scale):
    """
    Save altair plots to file, and create a new directory
    if file_path is not found
    
    Parameters
    ----------
    plot : Altair plot
        the plot to save
    file_path : str
        file path to save the plot
    scale : float
        the scale factor when saving the plot
    """
    try:
        plot.save(file_path, scale_factor=scale)
    except:
        os.makedirs(os.path.dirname(file_path))
        plot.save(file_path, scale_factor=scale)
  
if __name__ == "__main__":
    main(opt["--plot1_path"], opt["--plot2_path"], opt["--plot3_path"])
  
