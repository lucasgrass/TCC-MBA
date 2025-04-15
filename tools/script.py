import pandas as pd

import os

base_path = os.path.dirname(os.path.abspath(__file__))

path_csv_train = "/train.csv"
path_csv_test = "/test.csv"

df = pd.read_csv(base_path + path_csv_train)

df