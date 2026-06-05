# Bài 1:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

for feature in features:

    plt.figure(figsize=(6,4))

    plt.hist(df[feature], bins=20)

    plt.title(f"Histogram của {feature}")

    plt.show()

plt.figure(figsize=(8,5))

for feature in features:

    sns.kdeplot(df[feature], label=feature, fill=True)

plt.legend()

plt.show()

plt.figure(figsize=(8,5))

sns.boxplot(data=df[features])

plt.show()

# Bài 2:

import yfinance as yf
import matplotlib.pyplot as plt

ticker = "AAPL"

stock = yf.download(
    ticker,
    start="2022-01-01",
    end="2024-01-01"
)

print(stock.head())

plt.figure(figsize=(12,5))

plt.plot(stock["Close"])

plt.title("Giá cổ phiếu AAPL")

plt.show()

stock["MA50"] = stock["Close"].rolling(window=50).mean()

stock["MA200"] = stock["Close"].rolling(window=200).mean()

plt.figure(figsize=(12,5))

plt.plot(stock["Close"], label="Close")

plt.plot(stock["MA50"], label="MA50")

plt.plot(stock["MA200"], label="MA200")

plt.legend()

plt.show()

# Bài 3:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Iris.csv")

features = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]

for feature in features:

    plt.figure(figsize=(6,4))

    sns.boxplot(y=df[feature])

    plt.title(feature)

    plt.show()

# Bài 4:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Iris.csv")

numeric_df = df.select_dtypes(include=["float64", "int64"])

corr_matrix = numeric_df.corr()

print(corr_matrix)

plt.figure(figsize=(8,6))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm"
)

plt.show()