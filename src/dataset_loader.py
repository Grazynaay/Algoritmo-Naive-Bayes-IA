import pandas as pd

def carregar_iris():
    df = pd.read_csv("data/iris/iris.csv")
    return df