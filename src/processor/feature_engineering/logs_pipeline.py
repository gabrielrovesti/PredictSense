"""
Pipeline per feature engineering su dati di log.
"""

import logging
import re
import numpy as np
from typing import Dict, List, Any

from src.processor.feature_engineering.pipeline import Pipeline


class LogsPipeline(Pipeline):
    """
    Pipeline per feature engineering su dati di log.
    
    Implementa trasformazioni specifiche per log applicativi.
    """
    
    def __init__(self):
        """Inizializza la pipeline per dati di log."""
        self.logger = logging.getLogger(__name__)
        
        # Stato interno per calcoli incrementali
        self.last_data = {}
        
        # Configurazioni
        self.error_keywords = [
            'error', 'exception', 'fail', 'failed', 'timeout', 
            'crash', 'critical', 'fatal', 'invalid'
        ]
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa i dati di log applicando le trasformazioni.
        
        Args:
            data: Dati grezzi da processare
        
        Returns:
            Dati processati con features estratte
        """
        try:
            # Estrai features dai log
            features = self._extract_log_features(data)
            
            # Aggiorna lo stato interno
            self.last_data = data.copy()
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error in logs pipeline: {str(e)}")
            # In caso di errore, restituisci un dizionario vuoto
            return {}
    
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Trasforma i dati senza stato interno.
        
        Args:
            data: Dati grezzi da trasformare
        
        Returns:
            Dati trasformati
        """
        # Per i log, process e transform fanno la stessa cosa
        # poiché non teniamo stato tra le chiamate
        return self._extract_log_features(data)
    
    def _extract_log_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estrae features dai dati di log.
        
        Args:
            data: Dati di log
        
        Returns:
            Features estratte
        """
        # Verifica se i dati contengono log entries
        if not data or 'entries' not in data or not isinstance(data['entries'], list):
            return {}
        
        entries = data['entries']
        features = {}
        
        # Numero totale di entries
        features['total_entries'] = len(entries)
        
        # Conteggio per livello di log
        level_counts = {'DEBUG': 0, 'INFO': 0, 'WARN': 0, 'ERROR': 0, 'FATAL': 0}
        for entry in entries:
            if 'level' in entry and entry['level'] in level_counts:
                level_counts[entry['level']] += 1
        
        # Aggiungi i conteggi alle features
        for level, count in level_counts.items():
            features[f'{level.lower()}_count'] = count
        
        # Calcola i rapporti tra livelli
        if features['total_entries'] > 0:
            for level, count in level_counts.items():
                features[f'{level.lower()}_ratio'] = count / features['total_entries']
        
        # Calcola errori per servizio
        service_error_counts = {}
        for entry in entries:
            if ('level' in entry and entry['level'] in ['ERROR', 'FATAL'] and 
                'service' in entry):
                service = entry['service']
                if service not in service_error_counts:
                    service_error_counts[service] = 0
                service_error_counts[service] += 1
        
        # Aggiungi conteggi errori per servizio
        for service, count in service_error_counts.items():
            features[f'errors_{service}'] = count
        
        # Analisi dei messaggi di errore
        error_message_features = self._analyze_error_messages(entries)
        features.update(error_message_features)
        
        # Calcola la frequenza di errori nel tempo
        if 'timestamp' in entries[0]:
            error_frequency = self._calculate_error_frequency(entries)
            features.update(error_frequency)
        
        return features
    
    def _analyze_error_messages(self, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analizza i messaggi di errore per estrarre features.
        
        Args:
            entries: Lista di log entries
        
        Returns:
            Features estratte dai messaggi di errore
        """
        features = {}
        
        # Conteggio keyword nei messaggi di errore
        error_keyword_counts = {keyword: 0 for keyword in self.error_keywords}
        
        # Conteggio pattern comuni di errore
        error_patterns = {
            'database': 0,
            'network': 0,
            'timeout': 0,
            'memory': 0,
            'permission': 0,
            'authentication': 0
        }
        
        for entry in entries:
            if 'level' in entry and entry['level'] in ['ERROR', 'FATAL'] and 'message' in entry:
                message = entry['message'].lower()
                
                # Controlla keywords
                for keyword in self.error_keywords:
                    if keyword in message:
                        error_keyword_counts[keyword] += 1
                
                # Controlla pattern
                if any(term in message for term in ['database', 'db', 'sql', 'query']):
                    error_patterns['database'] += 1
                
                if any(term in message for term in ['network', 'connection', 'socket']):
                    error_patterns['network'] += 1
                
                if any(term in message for term in ['timeout', 'timed out']):
                    error_patterns['timeout'] += 1
                
                if any(term in message for term in ['memory', 'heap', 'stack', 'out of memory']):
                    error_patterns['memory'] += 1
                
                if any(term in message for term in ['permission', 'access', 'denied', 'forbidden']):
                    error_patterns['permission'] += 1
                
                if any(term in message for term in ['auth', 'login', 'password', 'credential']):
                    error_patterns['authentication'] += 1
        
        # Aggiungi conteggi alle features
        for keyword, count in error_keyword_counts.items():
            if count > 0:
                features[f'keyword_{keyword}'] = count
        
        for pattern, count in error_patterns.items():
            if count > 0:
                features[f'pattern_{pattern}'] = count
        
        return features
    
    def _calculate_error_frequency(self, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calcola la frequenza di errori nel tempo.
        
        Args:
            entries: Lista di log entries
        
        Returns:
            Features sulla frequenza di errori
        """
        features = {}
        
        # Filtra solo entries con timestamp e level Error/Fatal
        error_entries = [
            entry for entry in entries 
            if 'timestamp' in entry and 'level' in entry and entry['level'] in ['ERROR', 'FATAL']
        ]
        
        if len(error_entries) <= 1:
            return features
        
        # Ordina per timestamp
        error_entries.sort(key=lambda e: e['timestamp'])
        
        # Calcola intervalli di tempo tra errori
        time_intervals = []
        for i in range(1, len(error_entries)):
            try:
                # Converte timestamp stringhe in oggetti datetime
                from datetime import datetime
                t1 = datetime.strptime(error_entries[i-1]['timestamp'], "%Y-%m-%d %H:%M:%S")
                t2 = datetime.strptime(error_entries[i]['timestamp'], "%Y-%m-%d %H:%M:%S")
                
                # Calcola differenza in secondi
                diff = (t2 - t1).total_seconds()
                time_intervals.append(diff)
            except (ValueError, TypeError):
                # Ignora errori di parsing timestamp
                continue
        
        if time_intervals:
            # Calcola statistiche sugli intervalli
            features['error_interval_mean'] = np.mean(time_intervals)
            features['error_interval_min'] = np.min(time_intervals)
            features['error_interval_max'] = np.max(time_intervals)
            
            # Calcola errori al minuto
            total_time = sum(time_intervals)
            if total_time > 0:
                errors_per_minute = (len(time_intervals) + 1) / (total_time / 60)
                features['errors_per_minute'] = errors_per_minute
        
        return features


