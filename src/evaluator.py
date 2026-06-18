import numpy as np
from collections import defaultdict

def Precisão(y_true, y_pred):
    """
    Calcular acurácia: (acertos / total) * 100
    """
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return (correct / total) * 100


def Matriz_de_validação(y_true, y_pred):
    """
    Gerar matriz de confusão.
    
    Returns:
        dicionário {(true_class, pred_class): count}
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    matrix = defaultdict(int)
    
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[(true_label, pred_label)] += 1
    
    return dict(matrix), sorted(classes)


def Impressao_da_matriz_Validação(y_true, y_pred, dataset_name=""):
    """
    Imprimir matriz de confusão formatada.
    """
    matrix, classes = Matriz_de_validação(y_true, y_pred)
    
    print(f"\n{'='*60}")
    print(f"Matriz de Confusão - {dataset_name}")
    print(f"{'='*60}")
    
    # Header
    header = "Verdadeiro \\ Predito"
    for pred_class in classes:
        header += f"\t{pred_class}"
    print(header)
    
    # Linhas
    for true_class in classes:
        row = f"{true_class}"
        for pred_class in classes:
            count = matrix.get((true_class, pred_class), 0)
            row += f"\t{count}"
        print(row)
    print()


def Resultados(dataset_name, y_test, y_pred, train_time, pred_time):
    """
    Imprimir resultados de classificação.
    
    Args:
        dataset_name: nome do dataset (ex: "IRIS" ou "BREAST CANCER")
        y_test: rótulos verdadeiros
        y_pred: rótulos preditos
        train_time: tempo de treino em segundos
        pred_time: tempo de predição em segundos
    """
    acc = Precisão(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print(f"RESULTADOS - {dataset_name}")
    print(f"{'='*60}")
    print(f"Total de amostras testadas: {len(y_test)}")
    print(f"Acertos: {np.sum(y_test == y_pred)}")
    print(f"Taxa de acerto: {acc:.2f}%")
    print(f"Tempo de treino: {train_time:.6f}s")
    print(f"Tempo de classificação: {pred_time:.6f}s")
    
    # Matriz de confusão
    Impressao_da_matriz_Validação(y_test, y_pred, dataset_name)
