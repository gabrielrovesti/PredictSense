import tensorflow as tf
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from ..models.autoencoder import AnomalyAutoencoder
from ..models.lstm import TimeSeriesAnomalyDetector


class AnomalyEnsemble:
    """
    Ensemble di modelli per rilevamento anomalie.
    
    Combina i risultati di diversi modelli di anomaly detection
    per ottenere previsioni più robuste.
    """
    
    def __init__(
        self,
        models: List,
        weights: Optional[List[float]] = None,
        voting: str = 'soft'
    ):
        """
        Inizializza l'ensemble.
        
        Args:
            models: Lista di modelli da combinare
            weights: Pesi da assegnare a ciascun modello (se None, pesi uguali)
            voting: Modalità di voting ('hard' o 'soft')
        """
        self.models = models
        
        # Se i pesi non sono specificati, usa pesi uguali
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            # Normalizza i pesi
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        self.voting = voting
        self.threshold = 0.5  # Soglia per il voto finale
    
    def detect_anomalies(self, data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rileva anomalie combinando i risultati dei modelli.
        
        Args:
            data: Dizionario con i dati di input per ciascun modello
                 {'model_key': model_data}
        
        Returns:
            Tupla di (anomaly_mask, anomaly_scores)
        """
        model_results = []
        
        # Raccoglie i risultati di ogni modello
        for i, model in enumerate(self.models):
            model_key = model.name
            if isinstance(model, AnomalyAutoencoder):
                anomalies, scores = model.detect_anomalies(data[model_key])
            elif isinstance(model, TimeSeriesAnomalyDetector):
                x_key = f"{model_key}_x"
                y_key = f"{model_key}_y"
                anomalies, scores = model.detect_anomalies(data[x_key], data[y_key])
            else:
                raise ValueError(f"Unsupported model type: {type(model)}")
            
            # Normalizza i punteggi in [0, 1]
            norm_scores = self._normalize_scores(scores)
            model_results.append((anomalies, norm_scores))
        
        # Combina i risultati in base alla modalità di voting
        if self.voting == 'hard':
            # Hard voting: conta quanti modelli classificano come anomalia
            final_scores = np.zeros(len(model_results[0][0]))
            for i, (anomalies, _) in enumerate(model_results):
                final_scores += anomalies.astype(float) * self.weights[i]
            
            # Un esempio è un'anomalia se il punteggio ponderato supera la soglia
            final_anomalies = final_scores > self.threshold
            
        else:  # soft voting
            # Soft voting: media ponderata dei punteggi di anomalia
            final_scores = np.zeros(len(model_results[0][1]))
            for i, (_, scores) in enumerate(model_results):
                final_scores += scores * self.weights[i]
            
            # Un esempio è un'anomalia se il punteggio medio supera la soglia
            final_anomalies = final_scores > self.threshold
        
        return final_anomalies, final_scores
    
    def _normalize_scores(self, scores):
        """Normalizza i punteggi di anomalia in [0, 1]."""
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        # Previene divisione per zero
        if max_score == min_score:
            return np.ones_like(scores)
        
        return (scores - min_score) / (max_score - min_score)
    
    def set_threshold(self, threshold):
        """Imposta la soglia per il voto finale."""
        self.threshold = threshold
    
    def save(self, path):
        """Salva l'ensemble e tutti i modelli componenti."""
        # Crea directory se non esiste
        import os
        os.makedirs(path, exist_ok=True)
        
        # Salva configurazione dell'ensemble
        config = {
            'weights': self.weights,
            'voting': self.voting,
            'threshold': self.threshold,
            'model_paths': []
        }
        
        # Salva ogni modello
        for i, model in enumerate(self.models):
            model_path = os.path.join(path, f"model_{i}")
            os.makedirs(model_path, exist_ok=True)
            
            if isinstance(model, (AnomalyAutoencoder, TimeSeriesAnomalyDetector)):
                model.save(model_path)
            else:
                raise ValueError(f"Unsupported model type for saving: {type(model)}")
            
            config['model_paths'].append({
                'path': f"model_{i}",
                'type': model.__class__.__name__
            })
        
        # Salva configurazione
        import json
        with open(os.path.join(path, "ensemble_config.json"), 'w') as f:
            json.dump(config, f)
    
    @classmethod
    def load(cls, path):
        """Carica l'ensemble e tutti i modelli componenti."""
        import os
        import json
        
        # Carica configurazione
        with open(os.path.join(path, "ensemble_config.json"), 'r') as f:
            config = json.load(f)
        
        models = []
        # Carica ogni modello
        for model_info in config['model_paths']:
            model_path = os.path.join(path, model_info['path'])
            
            if model_info['type'] == 'AnomalyAutoencoder':
                model = tf.keras.models.load_model(model_path)
            elif model_info['type'] == 'TimeSeriesAnomalyDetector':
                model = tf.keras.models.load_model(model_path)
            else:
                raise ValueError(f"Unsupported model type: {model_info['type']}")
            
            models.append(model)
        
        # Crea istanza dell'ensemble
        instance = cls(models, weights=config['weights'], voting=config['voting'])
        instance.threshold = config['threshold']
        
        return instance