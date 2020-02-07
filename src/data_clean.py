""" Script used to import CSV data from a publicly accessible URL or relative file path.

Usage:
  data_clean.py --read_path=<read_path> --write_path=<write_path>

Example:
  python data_clean.py --read_path=../data/raw_data.csv --write_path=../data/

"""
from docopt import docopt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

opt = docopt(__doc__)

def main(read_path, write_path):

    df = pd.read_csv(read_path)

    X = df.drop(["C_YEAR", "C_VEHS", "C_CONF", "V_ID", "V_YEAR",
                 "P_ID", "P_PSN", "P_ISEV", "P_USER", "C_CASE", "C_SEV", "P_SEX", "P_AGE", "P_SAFE"], axis=1)
    y = df["C_SEV"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=407)
    # clean the train data, dropping rows with unknown/string values and convert data frame to numeric type
    combined_train = pd.concat([X_train, y_train], axis=1)
    cleaned_train = combined_train.apply(pd.to_numeric, errors='coerce').dropna(axis=0).astype(int)
    # clean the test data, dropping rows with unknown/string values and convert data frame to numeric type
    combined_test = pd.concat([X_test, y_test], axis=1)
    cleaned_test = combined_test.apply(pd.to_numeric, errors='coerce').dropna(axis=0).astype(int)

    print("Saving files...")
    cleaned_train.to_csv(write_path +  "cleaned_train_data.csv")
    cleaned_test.to_csv(write_path + "cleaned_test_data.csv")

  #  assert main("../data/assert.html", "../results/") == "file path should end with .csv", 'It should raise a warning'

if __name__ == "__main__":
    main(opt["--read_path"], opt["--write_path"])
