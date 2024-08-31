import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def make_dataset():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df["target"] = iris.target

    train, test = train_test_split(df, test_size=0.2, random_state=42)

    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)


if __name__ == "__main__":
    make_dataset()
