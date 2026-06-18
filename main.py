import time
import numpy as np
from src.naive_bayes import NaiveBayes
from src.dataset_loader import (
    carregar_iris, 
    carregar_breast_cancer, 
    dividir_treino_teste
)
from src.evaluator import Resultados, Impressao_da_matriz_Validação, Precisão

# ============================================================================
# IMPLEMENTAÇÃO NAIVE BAYES - SEM SKLEARN
# ============================================================================

def testar_dataset(dataset_name, X, y):
    """
    Testar Naive Bayes em um dataset.
    
    Args:
        dataset_name: nome do dataset
        X: atributos (array 2D)
        y: classes (array 1D)
    """
    print(f"\n\n{'#'*60}")
    print(f"# TESTANDO: {dataset_name}")
    print(f"{'#'*60}")
    print(f"Amostras: {len(X)}, Atributos: {X.shape[1]}, Classes: {len(np.unique(y))}")
    
    # Dividir dados
    X_train, X_test, y_train, y_test = dividir_treino_teste(X, y, tamanho_teste=0.3)
    
    print(f"Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")
    
    # ========== TREINAR ==========
    print("\n[1] Treinando modelo Naive Bayes...")
    nb = NaiveBayes()
    
    start = time.time()
    nb.treinar(X_train, y_train)
    train_time = time.time() - start
    
    print(f"✓ Modelo treinado em {train_time:.6f}s")
    
    # ========== PREDIZER ==========
    print("[2] Realizando predições...")
    
    start = time.time()
    y_pred = nb.predizer(X_test)
    pred_time = time.time() - start
    
    print(f"✓ Predições realizadas em {pred_time:.6f}s")
    
    # ========== AVALIAR ==========
    Resultados(dataset_name, y_test, y_pred, train_time, pred_time)
    
    return {
        'dataset': dataset_name,
        'accuracy': Precisão(y_test, y_pred),
        'train_time': train_time,
        'pred_time': pred_time
    }


def comparar_com_sklearn(dataset_name, X, y):
    """
    [EXTRA CREDIT] Comparar com sklearn GaussianNB
    """
    try:
        from sklearn.naive_bayes import GaussianNB
        from sklearn.model_selection import train_test_split
        
        print(f"\n\n{'*'*60}")
        print(f"* COMPARAÇÃO COM SKLEARN - {dataset_name}")
        print(f"{'*'*60}")
        
        # Dividir dados (mesma estratégia)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Treinar sklearn
        start = time.time()
        model_sklearn = GaussianNB()
        model_sklearn.fit(X_train, y_train)
        train_time_sklearn = time.time() - start
        
        # Predizer
        start = time.time()
        y_pred_sklearn = model_sklearn.predict(X_test)
        pred_time_sklearn = time.time() - start
        
        acc_sklearn = Precisão(y_test, y_pred_sklearn)
        
        print(f"Sklearn GaussianNB:")
        print(f"  Taxa de acerto: {acc_sklearn:.2f}%")
        print(f"  Tempo de treino: {train_time_sklearn:.6f}s")
        print(f"  Tempo de predição: {pred_time_sklearn:.6f}s")
        
        return {
            'dataset': dataset_name,
            'accuracy': acc_sklearn,
            'train_time': train_time_sklearn,
            'pred_time': pred_time_sklearn
        }
    except ImportError:
        print("⚠ Sklearn não instalado. Pulando comparação.")
        return None


def main():
    """Executar pipeline completo"""
    
    print("\n" + "="*60)
    print(" NAIVE BAYES - IMPLEMENTAÇÃO MANUAL")
    print("="*60)
    
    resultados = []
    
    # ===== TESTE 1: IRIS =====
    print("\n>>> Carregando IRIS...")
    df_iris = carregar_iris()
    X_iris = df_iris.iloc[:, :-1].values
    y_iris = df_iris.iloc[:, -1].values
    
    result_iris = testar_dataset("IRIS", X_iris, y_iris)
    resultados.append(result_iris)
    
    # ===== TESTE 2: BREAST CANCER =====
    print("\n>>> Carregando BREAST CANCER...")
    X_bc, y_bc = carregar_breast_cancer()
    
    result_bc = testar_dataset("BREAST CANCER (WDBC)", X_bc, y_bc)
    resultados.append(result_bc)
    
    # ===== COMPARAÇÃO COM SKLEARN (EXTRA CREDIT) =====
    print("\n\n" + "="*60)
    print(" EXTRA CREDIT - COMPARAÇÃO COM SKLEARN")
    print("="*60)
    
    result_sklearn_iris = comparar_com_sklearn("IRIS", X_iris, y_iris)
    result_sklearn_bc = comparar_com_sklearn("BREAST CANCER (WDBC)", X_bc, y_bc)
    
    # ===== RESUMO FINAL =====
    print("\n\n" + "="*60)
    print(" RESUMO FINAL")
    print("="*60)
    print("\nSeu Algoritmo (Naive Bayes Manual):")
    print("-" * 60)
    for r in resultados:
        print(f"{r['dataset']:25} | Acuracia: {r['accuracy']:6.2f}% | "
              f"Treino: {r['train_time']:.6f}s | Pred: {r['pred_time']:.6f}s")
    
    if result_sklearn_iris or result_sklearn_bc:
        print("\nSklearn GaussianNB:")
        print("-" * 60)
        if result_sklearn_iris:
            r = result_sklearn_iris
            print(f"{r['dataset']:25} | Acuracia: {r['accuracy']:6.2f}% | "
                  f"Treino: {r['train_time']:.6f}s | Pred: {r['pred_time']:.6f}s")
        if result_sklearn_bc:
            r = result_sklearn_bc
            print(f"{r['dataset']:25} | Acuracia: {r['accuracy']:6.2f}% | "
                  f"Treino: {r['train_time']:.6f}s | Pred: {r['pred_time']:.6f}s")


if __name__ == "__main__":
    main()