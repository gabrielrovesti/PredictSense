"""
Implementazione di trasformatori per feature engineering.
"""

import numpy as np
from typing import Dict, List, Any


class NumericalNormalizer:
    """
    Normalizza valori numerici in un range specifico.
    
    Tipicamente normalizza tra 0 e 1, o -1 e 1.
    """
    
    def __init__(self, method: str = 'minmax', target_range: tuple = (0, 1)):
        """
        Inizializza il normalizzatore.
        
        Args:
            method: Metodo di normalizzazione ('minmax', 'zscore')
            target_range: Range target per normalizzazione minmax
        """
        self.method = method
        self.target_range = target_range
        
        # Statistiche per ogni feature
        self.stats = {}
    
    def fit(self, data: Dict[str, float]):
        """
        Calcola le statistiche dai dati.
        
        Args:
            data: Dizionario di valori numerici
        """
        for feature, value in data.items():
            if feature not in self.stats:
                self.stats[feature] = {'min': float('inf'), 'max': float('-inf'), 
                                      'sum': 0, 'sum_sq': 0, 'count': 0}
            
            # Aggiorna min/max
            self.stats[feature]['min'] = min(self.stats[feature]['min'], value)
            self.stats[feature]['max'] = max(self.stats[feature]['max'], value)
            
            # Aggiorna statistiche per z-score
            self.stats[feature]['sum'] += value
            self.stats[feature]['sum_sq'] += value ** 2
            self.stats[feature]['count'] += 1
    
    def transform(self, data: Dict[str, float]) -> Dict[str, float]:
        """
        Normalizza i dati.
        
        Args:
            data: Dizionario di valori numerici
        
        Returns:
            Dizionario di valori normalizzati
        """
        # Aggiorna le statistiche con i nuovi dati
        self.fit(data)
        
        result = {}
        for feature, value in data.items():
            if feature in self.stats:
                if self.method == 'minmax':
                    result[feature] = self._minmax_normalize(feature, value)
                elif self.method == 'zscore':
                    result[feature] = self._zscore_normalize(feature, value)
                else:
                    # Se metodo non riconosciuto, mantieni valore originale
                    result[feature] = value
            else:
                # Se feature non ha statistiche, mantieni valore originale
                result[feature] = value
        
        return result
    
    def _minmax_normalize(self, feature: str, value: float) -> float:
        """
        Normalizza un valore usando min-max scaling.
        
        Args:
            feature: Nome della feature
            value: Valore da normalizzare
        
        Returns:
            Valore normalizzato
        """
        min_val = self.stats[feature]['min']
        max_val = self.stats[feature]['max']
        
        # Evita divisione per zero
        if max_val == min_val:
            return 0.5  # Valore medio del range target di default
        
        # Normalizza tra 0 e 1
        normalized = (value - min_val) / (max_val - min_val)
        
        # Scala al range target se diverso da (0,1)
        if self.target_range != (0, 1):
            target_min, target_max = self.target_range
            normalized = normalized * (target_max - target_min) + target_min
        
        return normalized
    
    def _zscore_normalize(self, feature: str, value: float) -> float:
        """
        Normalizza un valore usando z-score (standardizzazione).
        
        Args:
            feature: Nome della feature
            value: Valore da normalizzare
        
        Returns:
            Valore normalizzato
        """
        count = self.stats[feature]['count']
        if count <= 1:
            return 0.0  # Non abbastanza dati per calcolare z-score
        
        mean = self.stats[feature]['sum'] / count
        
        # Calcola deviazione standard
        variance = (self.stats[feature]['sum_sq'] / count) - (mean ** 2)
        # Evita errori numerici
        variance = max(variance, 1e-10)
        std_dev = np.sqrt(variance)
        
        # Calcola z-score
        return (value - mean) / std_dev


class MissingValueImputer:
    """
    Imputa valori mancanti nei dati.
    """
    
    def __init__(self, strategy: str = 'mean'):
        """
        Inizializza l'imputer.
        
        Args:
            strategy: Strategia di imputazione ('mean', 'median', 'mode')
        """
        self.strategy = strategy
        
        # Statistiche per ogni feature
        self.stats = {}
    
    def fit(self, data: Dict[str, Any]):
        """
        Calcola le statistiche dai dati.
        
        Args:
            data: Dizionario di valori
        """
        for feature, value in data.items():
            # Ignora valori non numerici o nulli
            if not isinstance(value, (int, float)) or value is None:
                continue
                
            if feature not in self.stats:
                self.stats[feature] = {'values': [], 'sum': 0, 'count': 0}
            
            self.stats[feature]['values'].append(value)
            self.stats[feature]['sum'] += value
            self.stats[feature]['count'] += 1
    
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Imputa valori mancanti nei dati.
        
        Args:
            data: Dizionario di valori con possibili null
        
        Returns:
            Dizionario con valori imputati
        """
        # Aggiorna le statistiche con i nuovi dati validi
        clean_data = {k: v for k, v in data.items() if v is not None and isinstance(v, (int, float))}
        self.fit(clean_data)
        
        result = {}
        for feature, value in data.items():
            if value is None and feature in self.stats:
                result[feature] = self._get_imputation_value(feature)
            else:
                result[feature] = value
        
        return result
    
    def _get_imputation_value(self, feature: str) -> float:
        """
        Calcola il valore di imputazione per una feature.
        
        Args:
            feature: Nome della feature
        
        Returns:
            Valore di imputazione
        """
        if feature not in self.stats or self.stats[feature]['count'] == 0:
            return 0.0  # Default se non ci sono statistiche
        
        if self.strategy == 'mean':
            return self.stats[feature]['sum'] / self.stats[feature]['count']
        
        elif self.strategy == 'median':
            values = sorted(self.stats[feature]['values'])
            middle = len(values) // 2
            
            if len(values) % 2 == 0:
                return (values[middle - 1] + values[middle]) / 2
            else:
                return values[middle]
        
        elif self.strategy == 'mode':
            from collections import Counter
            values = self.stats[feature]['values']
            return Counter(values).most_common(1)[0][0]
        
        else:
            # Strategia non riconosciuta, usa la media
            return self.stats[feature]['sum'] / self.stats[feature]['count']


class OutlierRemover:
    """
    Rimuove o sostituisce outlier nei dati.
    """
    
    def __init__(self, method: str = 'iqr', action: str = 'clip'):
        """
        Inizializza il rilevatore di outlier.
        
        Args:
            method: Metodo di rilevamento ('iqr', 'zscore')
            action: Azione da compiere ('clip', 'remove')
        """
        self.method = method
        self.action = action
        
        # Statistiche per ogni feature
        self.stats = {}
    
    def fit(self, data: Dict[str, float]):
        """
        Calcola le statistiche dai dati.
        
        Args:
            data: Dizionario di valori numerici
        """
        for feature, value in data.items():
            if not isinstance(value, (int, float)):
                continue
                
            if feature not in self.stats:
                self.stats[feature] = {'values': [], 'sum': 0, 'sum_sq': 0, 'count': 0}
            
            self.stats[feature]['values'].append(value)
            self.stats[feature]['sum'] += value
            self.stats[feature]['sum_sq'] += value ** 2
            self.stats[feature]['count'] += 1
    
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rimuove o sostituisce outlier nei dati.
        
        Args:
            data: Dizionario di valori numerici
        
        Returns:
            Dizionario con outlier trattati
        """
        # Aggiorna le statistiche con i nuovi dati
        clean_data = {k: v for k, v in data.items() if isinstance(v, (int, float))}
        self.fit(clean_data)
        
        result = {}
        for feature, value in data.items():
            if not isinstance(value, (int, float)):
                result[feature] = value
                continue
                
            if feature in self.stats:
                if self.method == 'iqr':
                    result[feature] = self._handle_iqr_outlier(feature, value)
                elif self.method == 'zscore':
                    result[feature] = self._handle_zscore_outlier(feature, value)
                else:
                    # Metodo non riconosciuto, mantieni valore originale
                    result[feature] = value
            else:
                # Feature senza statistiche, mantieni valore originale
                result[feature] = value
        
        return result
    
    def _handle_iqr_outlier(self, feature: str, value: float) -> float:
        """
        Gestisce outlier usando il metodo IQR.
        
        Args:
            feature: Nome della feature
            value: Valore da controllare
        
        Returns:
            Valore originale o corretto
        """
        if len(self.stats[feature]['values']) < 4:
            return value  # Non abbastanza dati per calcolare IQR
        
        values = sorted(self.stats[feature]['values'])
        n = len(values)
        
        # Calcola primo e terzo quartile
        q1_idx = n // 4
        q3_idx = (3 * n) // 4
        
        q1 = values[q1_idx]
        q3 = values[q3_idx]
        
        # Calcola IQR e limiti
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Controlla se è outlier
        if value < lower_bound or value > upper_bound:
            if self.action == 'clip':
                # Limita al bound più vicino
                return max(lower_bound, min(value, upper_bound))
            elif self.action == 'remove':
                # Ritorna None per rimuovere
                return None
        
        return value
    
    def _handle_zscore_outlier(self, feature: str, value: float) -> float:
        """
        Gestisce outlier usando il metodo Z-score.
        
        Args:
            feature: Nome della feature
            value: Valore da controllare
        
        Returns:
            Valore originale o corretto
        """
        count = self.stats[feature]['count']
        if count <= 1:
            return value  # Non abbastanza dati per calcolare z-score
        
        mean = self.stats[feature]['sum'] / count
        
        # Calcola deviazione standard
        variance = (self.stats[feature]['sum_sq'] / count) - (mean ** 2)
        # Evita errori numerici
        variance = max(variance, 1e-10)
        std_dev = np.sqrt(variance)
        
        # Calcola z-score
        z_score = (value - mean) / std_dev
        
        # Controlla se è outlier (|z| > 3)
        if abs(z_score) > 3:
            if self.action == 'clip':
                # Limita a +/-3 deviazioni standard
                if z_score > 0:
                    return mean + 3 * std_dev
                else:
                    return mean - 3 * std_dev
            elif self.action == 'remove':
                # Ritorna None per rimuovere
                return None
        
        return value