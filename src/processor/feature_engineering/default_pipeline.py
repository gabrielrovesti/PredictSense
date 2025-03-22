"""
Pipeline di default per feature engineering.
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


class DefaultPipeline(Pipeline):
    """
    Pipeline di default per feature engineering.
    
    Implementa trasformazioni generiche che possono essere applicate
    a vari tipi di dati quando non è disponibile una pipeline specifica.
    """
    
    def __init__(self):
        """Inizializza la pipeline di default."""
        self.logger = logging.getLogger(__name__)
        
        # Trasformatori
        self.normalizer = NumericalNormalizer()
        self.imputer = MissingValueImputer()
        self.outlier_remover = OutlierRemover()
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa i dati applicando le trasformazioni.
        
        Args:
            data: Dati grezzi da processare
        
        Returns:
            Dati processati con features estratte
        """
        try:
            # Applica le trasformazioni
            processed = self.transform(data)
            return processed
            
        except Exception as e:
            self.logger.error(f"Error in default pipeline: {str(e)}")
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
