# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os

from tools.utils import drop_outliers

base_path = os.path.dirname(os.path.abspath(__file__))

path_csv_train = "../datasets/train.csv"
path_csv_test = "../datasets/test.csv"

train_file_path = os.path.join(base_path, path_csv_train)
test_file_path = os.path.join(base_path, path_csv_test)

df = pd.read_csv(train_file_path, sep=",", decimal=".")

df
# %%

pd.set_option('display.max_columns', None)

print(df['Churn'].value_counts())


# %% Unique identifiers do not impact model predictions 
df.drop('CustomerID', axis=1, inplace=True)

# %% Check if there is null values
print("\n null values:", df.isnull().sum())
df.dropna(inplace=True)

# %% Check if there is duplicate lines
print("\nDuplicated lines:", df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)

# %%
print("\nDescriptive statistics:", df.describe())

numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
   sns.boxplot(df['col'])

# %% Removing outliers

numeric_cols = df.select_dtypes(include=[np.number]).columns
num_plots = len(numeric_cols)
fig, axs = plt.subplots(num_plots, 1, dpi=95, figsize=(7, num_plots * 2))

for i, col in enumerate(numeric_cols):
    axs[i].boxplot(df[col], vert=False)
    axs[i].set_ylabel(col)
plt.tight_layout()
plt.show()

print("\nBefore outliers function:", df.shape)

df = drop_outliers(df, "AccountAge")

#for col in numeric_cols:
#   df = drop_outliers(df, col)

print("\nAfter outliers function:", df.shape)

# %% Applying LabelEncoder and One-Hot Encoding

# LabelEncoder: DescriptionType
    
# One-Hot Encoding: PaymentMethod, PaperLessBiling, ContentType, MultiDeviceAccess
# DeviceRegistered, GenrePreference, Gender, ParentalControl, SubtitlesEnabled