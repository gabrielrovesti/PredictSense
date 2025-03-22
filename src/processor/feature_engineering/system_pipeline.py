"""
Pipeline per feature engineering su dati di sistema.
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


class SystemPipeline(Pipeline):
    """
    Pipeline per feature engineering su dati di sistema.
    
    Implementa trasformazioni specifiche per metriche di sistema.
    """
    
    def __init__(self):
        """Inizializza la pipeline per dati di sistema."""
        self.logger = logging.getLogger(__name__)
        
        # Stato interno per calcoli incrementali
        self.last_data = {}
        
        # Trasformatori
        self.normalizer = NumericalNormalizer()
        self.imputer = MissingValueImputer()
        self.outlier_remover = OutlierRemover()
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa i dati di sistema applicando le trasformazioni.
        
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
            self.logger.error(f"Error in system pipeline: {str(e)}")
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
        Estrae features aggiuntive dai dati di sistema.
        
        Args:
            data: Dati grezzi
        
        Returns:
            Dati con features aggiuntive
        """
        result = data.copy()
        
        # Calcola il carico complessivo del sistema
        if all(key in data for key in ['cpu_usage', 'memory_usage', 'io_wait']):
            # Peso maggiore a CPU e memoria rispetto a IO
            result['system_load_score'] = (0.5 * data['cpu_usage'] + 
                                          0.3 * data['memory_usage'] + 
                                          0.2 * data['io_wait'])
        
        # Calcola il rapporto tra CPU e IO
        if all(key in data for key in ['cpu_usage', 'io_wait']) and data['io_wait'] > 0:
            result['cpu_io_ratio'] = data['cpu_usage'] / data['io_wait']
        
        # Converti load_average in feature utilizzabile (se presente)
        if 'load_average' in data and isinstance(data['load_average'], list):
            # Estrai valori individuali
            if len(data['load_average']) >= 1:
                result['load_1m'] = data['load_average'][0]
            if len(data['load_average']) >= 2:
                result['load_5m'] = data['load_average'][1]
            if len(data['load_average']) >= 3:
                result['load_15m'] = data['load_average'][2]
            
            # Calcola trend del carico (positivo = in aumento, negativo = in diminuzione)
            if len(data['load_average']) >= 2:
                result['load_trend_short'] = data['load_average'][0] - data['load_average'][1]
            if len(data['load_average']) >= 3:
                result['load_trend_long'] = data['load_average'][0] - data['load_average'][2]
        
        # Calcolo delta rispetto all'ultima misurazione
        if self.last_data:
            for key in ['cpu_usage', 'memory_usage', 'disk_usage']:
                if key in data and key in self.last_data:
                    result[f'{key}_delta'] = data[key] - self.last_data[key]
        
        return result
