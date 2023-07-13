# Author: Amelia Tang 
# Project: Car price prediction

FROM jupyter/minimal-notebook

# Install R packages 
RUN conda install --quiet --yes \
    'r-base=4.2.3' \
    'r-knitr' \
    'r-kableextra' \
    'r-tidyverse' \
    'r-rmarkdown' \
    'r-scales' \
    'r-htmltools' \
    'r-htmlwidgets' 


# Install Python Packages (Conda)
RUN conda install --quiet --yes \
    'pandas' \
    'numpy' \
    'scikit-learn' \
    'matplotlib' \
    'shap' 
    
    
    
 RUN pip install \
     'docopt==0.6.2' \
     'xgboost' \
     'vega' 
     
