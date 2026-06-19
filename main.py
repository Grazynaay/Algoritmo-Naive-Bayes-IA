import time
import numpy as np
from src.naive_bayes import NaiveBayes
from src.dataset_loader import (
    carregar_iris, 
    carregar_breast_cancer, 
    dividir_treino_teste
)
from src.evaluator import Resultados, Impressao_da_matriz_Validação, Precisão
from src.visualizer import NaiveBayesVisualizer

# ============================================================================
# IMPLEMENTAÇÃO NAIVE BAYES - SEM SKLEARN
# ============================================================================

def testar_dataset(nome_dataset, dados, rotulos):
    """
    Testar Naive Bayes em um dataset.
    
    Args:
        nome_dataset: nome do dataset
        dados: atributos (array 2D)
        rotulos: classes (array 1D)
    """
    print(f"\n\n{'#'*60}")
    print(f"# TESTANDO: {nome_dataset}")
    print(f"{'#'*60}")
    print(f"Amostras: {len(dados)}, Atributos: {dados.shape[1]}, Classes: {len(np.unique(rotulos))}")
    
    # Dividir dados
    dados_treino, dados_teste, rotulos_treino, rotulos_teste = dividir_treino_teste(dados, rotulos, tamanho_teste=0.3)
    
    print(f"Treino: {len(dados_treino)} amostras | Teste: {len(dados_teste)} amostras")
    
    # ========== TREINAR ==========
    print("\n[1] Treinando modelo Naive Bayes...")
    classificador = NaiveBayes()
    
    start = time.time()
    classificador.treinar(dados_treino, rotulos_treino)
    tempo_treino = time.time() - start
    
    print(f"✓ Modelo treinado em {tempo_treino:.6f}s")
    
    # ========== PREDIZER ==========
    print("[2] Realizando predições...")
    
    start = time.time()
    rotulos_preditos = classificador.predizer(dados_teste)
    tempo_predicao = time.time() - start
    
    print(f"✓ Predições realizadas em {tempo_predicao:.6f}s")
    
    # ========== AVALIAR ==========
    Resultados(nome_dataset, rotulos_teste, rotulos_preditos, tempo_treino, tempo_predicao)
    
    accuracy = Precisão(rotulos_teste, rotulos_preditos)
    
    return {
        'dataset': nome_dataset,
        'accuracy': accuracy,
        'train_time': tempo_treino,
        'pred_time': tempo_predicao,
        'X_test': dados_teste,
        'y_test': rotulos_teste,
        'y_pred': rotulos_preditos
    }


def comparar_com_sklearn(nome_dataset, dados, rotulos):
    """
    [EXTRA CREDIT] Comparar com sklearn GaussianNB
    """
    try:
        from sklearn.naive_bayes import GaussianNB
        from sklearn.model_selection import train_test_split
        
        print(f"\n\n{'*'*60}")
        print(f"* COMPARAÇÃO COM SKLEARN - {nome_dataset}")
        print(f"{'*'*60}")
        
        # Dividir dados (mesma estratégia)
        dados_treino, dados_teste, rotulos_treino, rotulos_teste = train_test_split(
            dados, rotulos, test_size=0.3, random_state=42
        )
        
        # Treinar sklearn
        start = time.time()
        modelo_sklearn = GaussianNB()
        modelo_sklearn.fit(dados_treino, rotulos_treino)
        tempo_treino_sklearn = time.time() - start
        
        # Predizer e calcular acurácia usando .score()
        start = time.time()
        acuracia_sklearn = modelo_sklearn.score(dados_teste, rotulos_teste) * 100
        tempo_predicao_sklearn = time.time() - start
        
        print(f"Sklearn GaussianNB:")
        print(f"  Taxa de acerto: {acuracia_sklearn:.2f}%")
        print(f"  Tempo de treino: {tempo_treino_sklearn:.6f}s")
        print(f"  Tempo de predição: {tempo_predicao_sklearn:.6f}s")
        
        return {
            'dataset': nome_dataset,
            'accuracy': acuracia_sklearn,
            'train_time': tempo_treino_sklearn,
            'pred_time': tempo_predicao_sklearn
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
    dados_iris_df = carregar_iris()
    dados_iris = dados_iris_df.iloc[:, :-1].values
    rotulos_iris = dados_iris_df.iloc[:, -1].values
    
    resultado_iris = testar_dataset("IRIS", dados_iris, rotulos_iris)
    resultados.append(resultado_iris)
    
    # ===== TESTE 2: BREAST CANCER =====
    print("\n>>> Carregando BREAST CANCER...")
    dados_cancer, rotulos_cancer = carregar_breast_cancer()
    
    resultado_cancer = testar_dataset("BREAST CANCER (WDBC)", dados_cancer, rotulos_cancer)
    resultados.append(resultado_cancer)
    
    # ===== COMPARAÇÃO COM SKLEARN (EXTRA CREDIT) =====
    print("\n\n" + "="*60)
    print(" EXTRA CREDIT - COMPARAÇÃO COM SKLEARN")
    print("="*60)
    
    resultado_sklearn_iris = comparar_com_sklearn("IRIS", dados_iris, rotulos_iris)
    resultado_sklearn_cancer = comparar_com_sklearn("BREAST CANCER (WDBC)", dados_cancer, rotulos_cancer)
    
    # ===== RESUMO FINAL =====
    print("\n\n" + "="*60)
    print(" RESUMO FINAL")
    print("="*60)
    print("\nSeu Algoritmo (Naive Bayes Manual):")
    print("-" * 60)
    for resultado in resultados:
        print(f"{resultado['dataset']:25} | Acuracia: {resultado['accuracy']:6.2f}% | "
              f"Treino: {resultado['train_time']:.6f}s | Pred: {resultado['pred_time']:.6f}s")
    
    if resultado_sklearn_iris or resultado_sklearn_cancer:
        print("\nSklearn GaussianNB:")
        print("-" * 60)
        if resultado_sklearn_iris:
            resultado = resultado_sklearn_iris
            print(f"{resultado['dataset']:25} | Acuracia: {resultado['accuracy']:6.2f}% | "
                  f"Treino: {resultado['train_time']:.6f}s | Pred: {resultado['pred_time']:.6f}s")
        if resultado_sklearn_cancer:
            resultado = resultado_sklearn_cancer
            print(f"{resultado['dataset']:25} | Acuracia: {resultado['accuracy']:6.2f}% | "
                  f"Treino: {resultado['train_time']:.6f}s | Pred: {resultado['pred_time']:.6f}s")
    
    # ===== VISUALIZAR RESULTADOS =====
    print("\n\n" + "="*60)
    print(" VISUALIZANDO RESULTADOS")
    print("="*60)
    
    visualizer = NaiveBayesVisualizer(figsize=(11, 9))
    
    # Plotar Iris
    print("\n>>> Gerando gráfico de classificação IRIS...")
    resultado_iris_plot = resultados[0]
    visualizer.plot_iris(
        resultado_iris_plot['X_test'],
        resultado_iris_plot['y_test'],
        resultado_iris_plot['y_pred'],
        resultado_iris_plot['accuracy'],
        resultado_iris_plot['train_time'],
        resultado_iris_plot['pred_time']
    )
    
    # Plotar Breast Cancer
    print(">>> Gerando gráfico de classificação BREAST CANCER...")
    resultado_cancer_plot = resultados[1]
    visualizer.plot_breast_cancer(
        resultado_cancer_plot['X_test'],
        resultado_cancer_plot['y_test'],
        resultado_cancer_plot['y_pred'],
        resultado_cancer_plot['accuracy'],
        resultado_cancer_plot['train_time'],
        resultado_cancer_plot['pred_time']
    )


if __name__ == "__main__":
    main()