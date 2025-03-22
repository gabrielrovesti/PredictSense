"""
Pipeline per feature engineering su dati di rete.
"""

import logging
import numpy as np
from typing import Dict, List, Any

from src.processor.feature_engineering.pipeline import Pipeline
from src.processor.feature_engineering.transformers import (
    NumericalNormalizer,
    MissingValueImputer,
    OutlierRemover
)


class NetworkPipeline(Pipeline):
    """
    Pipeline per feature engineering su dati di rete.
    
    Implementa trasformazioni specifiche per metriche di rete.
    """
    
    def __init__(self):
        """Inizializza la pipeline per dati di rete."""
        self.logger = logging.getLogger(__name__)
        
        # Stato interno per calcoli incrementali
        self.last_data = {}
        
        # Trasformatori
        self.normalizer = NumericalNormalizer()
        self.imputer = MissingValueImputer()
        self.outlier_remover = OutlierRemover()
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa i dati di rete applicando le trasformazioni.
        
        Args:
            data: Dati grezzi da processare
        
        Returns:
            Dati processati con features estratte
        """
        try:
            # Estrai features aggiuntive
            enhanced_data = self._extract_features(data)
            
            # Applica le trasformazioni standard
            processed = self.transform(enhanced_data)
            
            # Aggiorna lo stato interno
            self.last_data = data.copy()
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Error in network pipeline: {str(e)}")
            # In caso di errore, restituisci i dati originali
            return data
    
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Trasforma i dati senza stato interno.
        
        Args:
            data: Dati grezzi da trasformare
        
        Returns:
            Dati trasformati
        """
        # Copia i dati per non modificare l'originale
        result = data.copy()
        
        # Estrai solo i valori numerici
        numerical_values = {}
        for key, value in result.items():
            if isinstance(value, (int, float)) and key != 'timestamp':
                numerical_values[key] = value
        
        # Applica pulizia, se ci sono valori numerici
        if numerical_values:
            # Rimuovi outlier
            cleaned = self.outlier_remover.transform(numerical_values)
            
            # Imputa valori mancanti
            imputed = self.imputer.transform(cleaned)
            
            # Normalizza valori numerici
            normalized = self.normalizer.transform(imputed)
            
            # Aggiorna i valori nel risultato
            for key, value in normalized.items():
                result[key] = value
        
        return result
    
    def _extract_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estrae features aggiuntive dai dati di rete.
        
        Args:
            data: Dati grezzi
        
        Returns:
            Dati con features aggiuntive
        """
        result = data.copy()
        
        # Calcolo traffico totale
        if 'bytes_sent' in data and 'bytes_received' in data:
            result['total_traffic'] = data['bytes_sent'] + data['bytes_received']
        
        # Calcolo ratio di upload/download
        if 'bytes_sent' in data and 'bytes_received' in data and data['bytes_received'] > 0:
            result['upload_download_ratio'] = data['bytes_sent'] / data['bytes_received']
        
        # Calcolo delta rispetto all'ultima misurazione
        if self.last_data:
            for key in ['bytes_sent', 'bytes_received', 'connection_count']:
                if key in data and key in self.last_data:
                    result[f'{key}_delta'] = data[key] - self.last_data[key]
        
        # Feature composita per qualità generale della rete
        if 'latency' in data and 'packet_loss' in data:
            # Più basso è meglio (bassa latenza, bassa perdita pacchetti)
            result['network_quality_score'] = 1.0 / (1.0 + data['latency'] * data['packet_loss'])
        
        return result
