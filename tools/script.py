import pandas as pd
import os

base_path = os.path.dirname(os.path.abspath(__file__))

path_csv_train = "../datasets/train.csv"
path_csv_test = "../datasets/test.csv"

train_file_path = os.path.join(base_path, path_csv_train)
test_file_path = os.path.join(base_path, path_csv_test)

df = pd.read_csv(train_file_path)

df