import pandas as pd
import numpy as np

def carregar_iris():
    """Carregar dataset Iris"""
    df = pd.read_csv("data/iris/iris.csv")
    return df


def carregar_breast_cancer():
    """
    Carregar dataset Breast Cancer (WDBC).
    
    Formato do arquivo wdbc.data:
    - Coluna 0: ID (descartar)
    - Coluna 1: Diagnosis (M=Malignant, B=Benign)
    - Colunas 2-31: 30 atributos numéricos
    
    Returns:
        X: array numpy (569, 30) com atributos
        y: array numpy (569,) com classes (M ou B)
    """
    # Ler arquivo sem header
    df = pd.read_csv("data/breast_cancer/wdbc.data", header=None)
    
    # Coluna 1 é a classe, colunas 2+ são atributos
    X = df.iloc[:, 2:].values  # (569, 30)
    y = df.iloc[:, 1].values   # (569,) com valores 'M' ou 'B'
    
    return X, y


def dividir_treino_teste(X, y, tamanho_teste=0.3, estado_aleatorio=42):
    """
    Dividir dados em treino e teste.
    
    Args:
        X: array 2D de atributos
        y: array 1D de classes
        tamanho_teste: fração para teste (padrão 0.3 = 30%)
        estado_aleatorio: seed para reprodutibilidade
        
    Returns:
        X_treino, X_teste, y_treino, y_teste
    """
    np.random.seed(estado_aleatorio)
    
    # Embaralhar índices
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    # Dividir
    split_idx = int(len(X) * (1 - tamanho_teste))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]