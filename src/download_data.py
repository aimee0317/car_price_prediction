# Author: Amelia Tang
# date: 12/20/2021 

"""Downloads data from a url and save it to a local filepath as csv.

Usage: download_data.py --url=<url> --out_file=<out_file> 

Options:
--url=<url>   URL from which too download the data(must be standard CSV format)
--out_file=<out_file> Path includingthe file name of where to write the file locally

"""

import os 
import requests
from docopt import docopt 

opt = docopt(__doc__) 

def main(url, out_file):
  data = requests.get(url)
  try:
    open(out_file, "wb").write(data.content)
  except:
    os.makedirs(os.path.dirname(out_file))
    open(out_file, "wb").write(data.content)

if __name__ == "__main__":
  main(opt["--url"], opt["--out_file"])
