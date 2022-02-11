# Author: Amelia Tang

"""
EDA plots
Usage: EDA_export.py --plot1_path=<plot1_path> --plot2_path=<plot2_path> --plot3_path=<plot3_path>

Options:
--plot1_path=<plot1_path>             file path of the target_distribution_plot
--plot2_path=<plot2_path>             file path of the price_by_brand
--plot3_path=<plot3_path>             file path of the price_year
"""

# import packages
import os
import pandas as pd 
import numpy as np 
import altair as alt
from altair_saver import save
from docopt import docopt

opt = docopt(__doc__)

def main(plot1_path, plot2_path, plot3_path):
    X_train = pd.read_csv("data/raw/X_train.csv", parse_dates=['year'])
    X_train['year'] = X_train['year'].dt.year
    y_train = pd.read_csv("data/raw/y_train.csv")
    data = X_train.join(y_train.set_index('carID'), on = "carID")
  
    plot1 = target_distribution_plot(data)
    plot2 = price_by_brand(data)
    plot3 = price_year_brand(data)
  
    save_plot(plot1, plot1_path, 4)
    save_plot(plot2, plot2_path, 4)
    save_plot(plot3, plot3_path, 4)
    
def target_distribution_plot(data):
    plot1 = alt.Chart(data, title ="Used Car Price Distribution").mark_bar().encode(
        alt.X('price', bin=alt.Bin(maxbins=30), title="Price"),
        y='count()')
  
    return plot1
  
def price_by_brand(data):
    plot2 = alt.Chart(data, title="Price Distribution by Brand").mark_boxplot().encode(
      x='price',
      y=alt.Y('brand'),
      color='brand')
  
    return plot2

  
def price_year_brand(data):
    plot3 = alt.Chart(data).mark_boxplot().encode(
        y='price',
        x=alt.X('year', scale=alt.Scale(zero=False)),
        color=alt.Color('year', legend=None)).facet('brand', columns=3, title = 'Price Distribution by Year for Each Brand')
    
    return plot3
  
def save_plot(plot, file_path, scale):
    try:
        plot.save(file_path, scale_factor=scale)
    except:
        os.makedirs(os.path.dirname(file_path))
        plot.save(file_path, scale_factor=scale)

if __name__ == "__main__":
    main(opt["--plot1_path"], opt["--plot2_path"], opt["--plot3_path"])
