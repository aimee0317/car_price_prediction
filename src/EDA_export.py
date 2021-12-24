"""
EDA plots
Usage: EDA_export.py --plot1path=<plot1path> --plot2path=<plot2path> --plot3path=<plot3path> --plot4path=<plot4path>

Options:
--plot1path=<plot1path>              file path of target_distribution_plot
--plot2path=<plot2path>              file path of price_by_brand
--plot3path=<plot3path>              file path of corr_plot
--plot4path=<plot4path>              file path of price_year
"""


import os
import pandas as pd 
import numpy as np 
import altair as alt
from altair_saver import save
from docopt import docopt


opt = docopt(__doc__)

X_train = pd.read_csv("data/raw/X_train.csv", parse_dates=['year'])
X_train['year'] = X_train['year'].dt.year
y_train = pd.read_csv("data/raw/y_train.csv")
train_df = X_train.join(y_train.set_index('carID'), on = "carID")

def main(plot1path, plot2path, plot3path, plot4path):
  
  target_distribution_plot(plot1path)
  price_by_brand(plot2path)
  corr_plot(plot3path)
  price_year(plot4path)

    
def target_distribution_plot(plot1path):
  """
  Generate a histagram for the distribution of the targert column 
  
  Parameters:
  --------
  plot1path : str
    path to save the graph
    
  Returns:
  --------
  save the graph to a specified directory
   """
  plot1 = alt.Chart(train_df, title ="Used Car Price Distribution").mark_bar().encode(
    alt.X('price', bin=alt.Bin(maxbins=30), title="Price"),
    y='count()')
  
  try:
    plot1.save(plot1path, scale_factor=4)
  except:
    os.makedirs(os.path.dirname(plot1path))
    plot1.save(plot1path, scale_factor=4)
  
def price_by_brand(plot2path):
  """
  Generate boxplots to show the distributions of price by brand   
  
  Parameters:
  --------
  plot2path : str
    path to save the plot
    
  Returns:
  --------
  save the plot to a specified directory
   """
  plot2 = alt.Chart(train_df, title ="Scatter plot of Price vs. Year").mark_point().encode(
    alt.X('year',scale=alt.Scale(zero=False)),
    y='price')
  
  plot2.save(plot2path, scale_factor=4)


def corr_plot(plot3path):
  """
  Generate correlation plots among numeric variables   
  
  Parameters:
  --------
  plot3path : str
    path to save the plot
    
  Returns:
  --------
  save the plot to a specified directory
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
     
  plot3.save(plot3path, scale_factor=4) 
  
def price_year(plot4path):
    """
    Generate price distribution plot by year for each brand 
  
    Parameters:
    --------
    plot4path : str
      path to save the plot
    
    Returns:
    --------
    save the plot to a specified directory
    """
    plot4 = alt.Chart(train_df).mark_boxplot().encode(
     y='price',
     x=alt.X('year', scale=alt.Scale(zero=False)),
     color=alt.Color('year', legend=None)).facet('brand', columns=1, title = 'Price Distribution by Year for Each Brand')
    
    plot4.save(plot4path, scale_factor=4)
  
  
if __name__ == "__main__":
    main(opt["--plot1path"], opt["--plot2path"], opt["--plot3path"], opt["--plot4path"])
  
