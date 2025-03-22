import tensorflow as tf
from tensorflow.keras import layers, Model
import mlflow
import mlflow.tensorflow
from typing import Tuple, Dict, Any, Optional
import numpy as np


class AnomalyAutoencoder(Model):
    """
    Modello Autoencoder per rilevamento anomalie.
    
    Implementa un autoencoder con architettura personalizzabile per
    rilevare anomalie in dati multidimensionali.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        encoding_dim: int = 32,
        hidden_layers: Tuple[int, ...] = (128, 64),
        dropout_rate: float = 0.2,
        activation: str = 'relu',
        name: str = "anomaly_autoencoder"
    ):
        """
        Inizializza un modello autoencoder per anomaly detection.
        
        Args:
            input_shape: Forma dell'input (n_features,)
            encoding_dim: Dimensione dello spazio latente
            hidden_layers: Tuple di dimensioni per i layer nascosti
            dropout_rate: Tasso di dropout per regolarizzazione
            activation: Funzione di attivazione
            name: Nome del modello
        """
        super(AnomalyAutoencoder, self).__init__(name=name)
        
        # Parametri del modello
        self.input_shape = input_shape
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.activation = activation
        
        # Soglia di rilevamento anomalie (calcolata dopo il training)
        self.threshold = None
        
        # Layers dell'encoder
        self.encoder_layers = []
        for units in hidden_layers:
            self.encoder_layers.append(layers.Dense(units, activation=activation))
            self.encoder_layers.append(layers.Dropout(dropout_rate))
        self.encoder_layers.append(layers.Dense(encoding_dim, activation=activation))
        
        # Layers del decoder
        self.decoder_layers = []
        for units in reversed(hidden_layers):
            self.decoder_layers.append(layers.Dense(units, activation=activation))
            self.decoder_layers.append(layers.Dropout(dropout_rate))
        self.decoder_layers.append(layers.Dense(input_shape[0], activation='sigmoid'))
    
    def call(self, inputs, training=None):
        """Forward pass del modello."""
        x = inputs
        
        # Encoder
        for layer in self.encoder_layers:
            if isinstance(layer, layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        
        encoded = x
        
        # Decoder
        for layer in self.decoder_layers:
            if isinstance(layer, layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        
        return x
    
    def encode(self, inputs):
        """Codifica input nello spazio latente."""
        x = inputs
        for layer in self.encoder_layers:
            if not isinstance(layer, layers.Dropout):
                x = layer(x)
        return x
    
    def compute_reconstruction_error(self, inputs):
        """Calcola l'errore di ricostruzione per ogni esempio."""
        reconstructions = self.predict(inputs)
        errors = tf.reduce_mean(tf.square(inputs - reconstructions), axis=1)
        return errors
    
    def set_threshold(self, validation_data, contamination=0.01):
        """
        Imposta la soglia per rilevamento anomalie basandosi sui dati di validazione.
        
        Args:
            validation_data: Dataset di validazione
            contamination: Frazione prevista di anomalie nei dati
        """
        errors = self.compute_reconstruction_error(validation_data)
        # Imposta la soglia come il quantile (1-contamination)
        self.threshold = np.quantile(errors, 1 - contamination)
        return self.threshold
    
    def detect_anomalies(self, data):
        """
        Rileva anomalie nei dati.
        
        Args:
            data: Dati da analizzare
            
        Returns:
            Tupla di (anomaly_mask, anomaly_scores)
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call set_threshold first.")
        
        scores = self.compute_reconstruction_error(data)
        anomalies = scores > self.threshold
        
        return anomalies, scores
    
    def get_config(self):
        """Restituisce la configurazione del modello."""
        return {
            'input_shape': self.input_shape,
            'encoding_dim': self.encoding_dim,
            'hidden_layers': self.hidden_layers,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'threshold': self.threshold
        }
    
    @classmethod
    def from_config(cls, config):
        """Crea un'istanza da una configurazione."""
        instance = cls(
            input_shape=config['input_shape'],
            encoding_dim=config['encoding_dim'],
            hidden_layers=config['hidden_layers'],
            dropout_rate=config['dropout_rate'],
            activation=config['activation']
        )
        instance.threshold = config.get('threshold')
        return instance