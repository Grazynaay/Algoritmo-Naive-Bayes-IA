import numpy as np
import math

class NaiveBayes:
    """
    Classificador Naive Bayes com atributos numéricos usando distribuição Gaussiana.
    Implementação manual seguindo a fórmula do professor:
    P[classe|x] = P[x|classe] * P[classe] / P[x]
    """
    
    def __init__(self):
        self.classes = None
        self.priori_classes = {}  # P[classe]
        self.feature_stats = {}  # {classe: {'media': [...], 'desvio': [...]}}
    
    def treinar(self, dados, rotulos):
        """
        Treinar o modelo Naive Bayes.
        
        Args:
            dados: array 2D (n_amostras, n_atributos) com atributos numéricos
            rotulos: array 1D (n_amostras,) com rótulos de classe
        """
        dados = np.array(dados)
        rotulos = np.array(rotulos)
        
        self.classes = np.unique(rotulos)
        n_amostras, n_atributos = dados.shape
        
        # Para cada classe, calcular P[classe] e estatísticas dos atributos
        for classe in self.classes:
            dados_classe = dados[rotulos == classe]  # Amostras da classe
            
            # P[classe] = número de amostras da classe / total
            self.priori_classes[classe] = len(dados_classe) / n_amostras
            
            # Para cada atributo: média e desvio padrão
            self.feature_stats[classe] = {
                'media': dados_classe.mean(axis=0),
                'desvio': dados_classe.std(axis=0)
            }
    
    def _pdf_gaussiana(self, valor, media, desvio):
        """
        Calcular PDF gaussiana: P(valor|media, desvio)
        Fórmula: (1 / sqrt(2π*desvio²)) * exp(-(valor-media)² / (2*desvio²))
        """
        if desvio == 0:
            return 1.0
        
        numerador = math.exp(-(valor - media) ** 2 / (2 * desvio ** 2))
        denominador = math.sqrt(2 * math.pi * desvio ** 2)
        return numerador / denominador
    
    def predizer(self, dados):
        """
        Classificar amostras usando MAP (Máximo a Posteriori).
        
        Args:
            dados: array 2D (n_amostras, n_atributos) com amostras a classificar
            
        Returns:
            array 1D com classes preditas
        """
        dados = np.array(dados)
        predicoes = []
        
        for amostra in dados:
            # Para cada classe, calcular log(P[classe|amostra]) = log(P[classe]) + log(P[amostra|classe])
            posteriores = {}
            
            for classe in self.classes:
                # log(P[classe])
                priori = np.log(self.priori_classes[classe])
                
                # log(P[amostra|classe]) = Σ log(P[atributo|classe])
                verossimilhanca = 0
                for indice, atributo in enumerate(amostra):
                    media = self.feature_stats[classe]['media'][indice]
                    desvio = self.feature_stats[classe]['desvio'][indice]
                    verossimilhanca += np.log(self._pdf_gaussiana(atributo, media, desvio) + 1e-10)  # +eps evita log(0)
                
                # log(P[classe|amostra]) ∝ log(P[classe]) + log(P[amostra|classe])
                posteriores[classe] = priori + verossimilhanca
            
            # Retornar classe com maior probabilidade a posteriori
            predicoes.append(max(posteriores, key=posteriores.get))
        
        return np.array(predicoes)
