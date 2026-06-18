import numpy as np
import math

class NaiveBayes:
    """
    Classificador Naive Bayes com atributos numéricos usando distribuição Gaussiana.
    Implementação manual seguindo a fórmula do professor:
    P[c|x] = P[x|c] * P[c] / P[x]
    """
    
    def __init__(self):
        self.classes = None
        self.class_priors = {}  # P[c]
        self.feature_stats = {}  # {classe: {'mean': [...], 'std': [...]}}
    
    def treinar(self, X, y):
        """
        Treinar o modelo Naive Bayes.
        
        Args:
            X: array 2D (n_samples, n_features) com atributos numéricos
            y: array 1D (n_samples,) com rótulos de classe
        """
        X = np.array(X)
        y = np.array(y)
        
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        
        # Para cada classe, calcular P[c] e estatísticas dos atributos
        for c in self.classes:
            X_c = X[y == c]  # Amostras da classe c
            
            # P[c] = número de amostras da classe / total
            self.class_priors[c] = len(X_c) / n_samples
            
            # Para cada atributo: média e desvio padrão
            self.feature_stats[c] = {
                'mean': X_c.mean(axis=0),
                'std': X_c.std(axis=0)
            }
    
    def _pdf_gaussiana(self, x, mean, std):
        """
        Calcular PDF gaussiana: P(x|mean, std)
        Fórmula: (1 / sqrt(2π*std²)) * exp(-(x-mean)² / (2*std²))
        """
        if std == 0:
            return 1.0
        
        numerador = math.exp(-(x - mean) ** 2 / (2 * std ** 2))
        denominador = math.sqrt(2 * math.pi * std ** 2)
        return numerador / denominador
    
    def predizer(self, X):
        """
        Classificar amostras usando MAP (Maximum a Posteriori).
        
        Args:
            X: array 2D (n_samples, n_features) com amostras a classificar
            
        Returns:
            array 1D com classes preditas
        """
        X = np.array(X)
        predictions = []
        
        for x in X:
            # Para cada classe, calcular log(P[c|x]) = log(P[c]) + log(P[x|c])
            posteriors = {}
            
            for c in self.classes:
                # log(P[c])
                prior = np.log(self.class_priors[c])
                
                # log(P[x|c]) = Σ log(P[xi|c])
                likelihood = 0
                for i, x_i in enumerate(x):
                    mean = self.feature_stats[c]['mean'][i]
                    std = self.feature_stats[c]['std'][i]
                    likelihood += np.log(self._pdf_gaussiana(x_i, mean, std) + 1e-10)  # +eps evita log(0)
                
                # log(P[c|x]) ∝ log(P[c]) + log(P[x|c])
                posteriors[c] = prior + likelihood
            
            # Retornar classe com maior probabilidade a posteriori
            predictions.append(max(posteriors, key=posteriors.get))
        
        return np.array(predictions)
