# Bài 1:
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt

iris = load_iris(as_frame=True)
df = iris.frame

print(df.head(10))
print(df.shape)
print(df.dtypes)
print(df.describe())

df.hist(figsize=(10,8))
plt.show()

df['target'].value_counts().plot(kind='bar')
plt.show()

# Bài 2:

from sklearn.preprocessing import StandardScaler
import numpy as np

print(df.isnull().sum())

df.iloc[0,0] = np.nan

for col in df.select_dtypes(include='number').columns:
    df[col] = df[col].fillna(df[col].mean())

df = df.drop_duplicates()

scaler = StandardScaler()
df[df.select_dtypes(include='number').columns] = scaler.fit_transform(
    df.select_dtypes(include='number')
)

df.boxplot()
plt.show()