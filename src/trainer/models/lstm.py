import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Tuple, Dict, Any, Optional, List
import numpy as np


class TimeSeriesAnomalyDetector(Model):
    """
    Modello LSTM-GRU per rilevamento anomalie su serie temporali.
    
    Implementa un modello di previsione di serie temporali che può 
    essere utilizzato per rilevare anomalie in base all'errore di previsione.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int],  # (seq_length, n_features)
        output_dim: int,
        lstm_units: List[int] = [64, 32],
        gru_units: int = 16,
        dense_units: List[int] = [16, 8],
        dropout_rate: float = 0.2,
        name: str = "ts_anomaly_detector"
    ):
        """
        Inizializza il modello per anomaly detection su serie temporali.
        
        Args:
            input_shape: Forma dell'input (seq_length, n_features)
            output_dim: Dimensione dell'output
            lstm_units: Lista di unità per i layer LSTM
            gru_units: Unità per il layer GRU
            dense_units: Lista di unità per i layer Dense
            dropout_rate: Tasso di dropout
            name: Nome del modello
        """
        super(TimeSeriesAnomalyDetector, self).__init__(name=name)
        
        # Parametri del modello
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.lstm_units = lstm_units
        self.gru_units = gru_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        
        # Soglia di rilevamento anomalie
        self.threshold = None
        
        # Definizione del modello
        self.lstm_layers = []
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            self.lstm_layers.append(
                layers.LSTM(
                    units, 
                    return_sequences=return_sequences,
                    name=f"lstm_{i}"
                )
            )
            self.lstm_layers.append(layers.Dropout(dropout_rate))
        
        self.gru_layer = layers.GRU(gru_units, name="gru")
        self.dropout = layers.Dropout(dropout_rate)
        
        self.dense_layers = []
        for i, units in enumerate(dense_units):
            self.dense_layers.append(layers.Dense(units, activation='relu', name=f"dense_{i}"))
            self.dense_layers.append(layers.Dropout(dropout_rate))
        
        self.output_layer = layers.Dense(output_dim, name="output")
    
    def call(self, inputs, training=None):
        """Forward pass del modello."""
        x = inputs
        
        # LSTM layers
        for i, layer in enumerate(self.lstm_layers):
            if isinstance(layer, layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        
        # GRU layer
        x = self.gru_layer(x)
        x = self.dropout(x, training=training)
        
        # Dense layers
        for layer in self.dense_layers:
            if isinstance(layer, layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        
        # Output layer
        return self.output_layer(x)
    
    def compute_prediction_error(self, x, y_true):
        """Calcola l'errore di previsione."""
        y_pred = self.predict(x)
        errors = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
        return errors
    
    def set_threshold(self, validation_x, validation_y, contamination=0.01):
        """
        Imposta la soglia per rilevamento anomalie basandosi sui dati di validazione.
        
        Args:
            validation_x: Input di validazione
            validation_y: Target di validazione
            contamination: Frazione prevista di anomalie nei dati
        """
        errors = self.compute_prediction_error(validation_x, validation_y)
        # Imposta la soglia come il quantile (1-contamination)
        self.threshold = np.quantile(errors, 1 - contamination)
        return self.threshold
    
    def detect_anomalies(self, x, y_true):
        """
        Rileva anomalie confrontando previsioni con valori reali.
        
        Args:
            x: Dati di input
            y_true: Valori reali da confrontare con le previsioni
            
        Returns:
            Tupla di (anomaly_mask, anomaly_scores)
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call set_threshold first.")
        
        scores = self.compute_prediction_error(x, y_true)
        anomalies = scores > self.threshold
        
        return anomalies, scores
    
    def get_config(self):
        """Restituisce la configurazione del modello."""
        return {
            'input_shape': self.input_shape,
            'output_dim': self.output_dim,
            'lstm_units': self.lstm_units,
            'gru_units': self.gru_units,
            'dense_units': self.dense_units,
            'dropout_rate': self.dropout_rate,
            'threshold': self.threshold
        }
    
    @classmethod
    def from_config(cls, config):
        """Crea un'istanza da una configurazione."""
        instance = cls(
            input_shape=config['input_shape'],
            output_dim=config['output_dim'],
            lstm_units=config['lstm_units'],
            gru_units=config['gru_units'],
            dense_units=config['dense_units'],
            dropout_rate=config['dropout_rate']
        )
        instance.threshold = config.get('threshold')
        return instance