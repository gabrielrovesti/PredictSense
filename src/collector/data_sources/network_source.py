"""
Fonte dati di rete per il collector.
Raccoglie metriche di rete come traffico, latenza, ecc.
"""

import os
import logging
import asyncio
import socket
import time
import json
import random
from typing import Dict, List, Any, Optional


class NetworkDataSource:
    """
    Fonte dati per metriche di rete.
    
    Raccoglie metriche come:
    - Byte inviati/ricevuti
    - Perdita di pacchetti
    - Latenza
    - Numero di connessioni
    """
    
    def __init__(self, source_id: str, config: Dict[str, Any]):
        """
        Inizializza la fonte dati di rete.
        
        Args:
            source_id: Identificativo della fonte dati
            config: Configurazione della fonte dati
        """
        self.source_id = source_id
        self.config = config if isinstance(config, dict) else json.loads(config)
        self.logger = logging.getLogger(__name__)
        
        # Configura metriche da raccogliere
        self.metrics = self.config.get("metrics", [
            "bytes_sent", "bytes_received", "packet_loss", "latency", "connection_count"
        ])
        
        # Configurazione del polling
        self.collection_interval = self.config.get("collection_interval", 60)
        
        # Stato interno
        self.last_bytes_sent = 0
        self.last_bytes_received = 0
    
    async def collect(self) -> Dict[str, Any]:
        """
        Raccoglie le metriche di rete.
        
        In un'implementazione reale, questo metodo raccoglierebbe 
        effettivamente i dati dalla rete. Per dimostrazione, genera dati simulati.
        
        Returns:
            Dizionario di metriche raccolte
        """
        try:
            # Simulazione raccolta dati
            # In un'implementazione reale, usare librerie come psutil o subprocess
            # per raccogliere effettivamente i dati di rete
            
            result = {
                "timestamp": time.time()
            }
            
            # Raccoglie metriche richieste
            if "bytes_sent" in self.metrics:
                bytes_sent = self._get_simulated_bytes_sent()
                result["bytes_sent"] = bytes_sent
                
                # Calcola delta rispetto all'ultima lettura
                if self.last_bytes_sent > 0:
                    result["bytes_sent_delta"] = bytes_sent - self.last_bytes_sent
                
                self.last_bytes_sent = bytes_sent
            
            if "bytes_received" in self.metrics:
                bytes_received = self._get_simulated_bytes_received()
                result["bytes_received"] = bytes_received
                
                # Calcola delta rispetto all'ultima lettura
                if self.last_bytes_received > 0:
                    result["bytes_received_delta"] = bytes_received - self.last_bytes_received
                
                self.last_bytes_received = bytes_received
            
            if "packet_loss" in self.metrics:
                result["packet_loss"] = self._get_simulated_packet_loss()
            
            if "latency" in self.metrics:
                result["latency"] = self._get_simulated_latency()
            
            if "connection_count" in self.metrics:
                result["connection_count"] = self._get_simulated_connection_count()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error collecting network metrics: {str(e)}")
            return {}
    
    def _get_simulated_bytes_sent(self) -> int:
        """Simula il conteggio di byte inviati."""
        # Assume una crescita costante più rumore casuale
        if self.last_bytes_sent == 0:
            # Prima lettura
            return random.randint(1000000, 2000000)
        else:
            # Aumenta dell'1-5% rispetto all'ultimo valore più variazione casuale
            growth = random.uniform(0.01, 0.05)
            variation = random.uniform(-0.01, 0.02)
            new_value = self.last_bytes_sent * (1 + growth + variation)
            return int(new_value)
    
    def _get_simulated_bytes_received(self) -> int:
        """Simula il conteggio di byte ricevuti."""
        # Simile ai byte inviati ma leggermente diverso
        if self.last_bytes_received == 0:
            # Prima lettura
            return random.randint(2000000, 3000000)
        else:
            # Aumenta dell'1-7% rispetto all'ultimo valore più variazione casuale
            growth = random.uniform(0.01, 0.07)
            variation = random.uniform(-0.015, 0.025)
            new_value = self.last_bytes_received * (1 + growth + variation)
            return int(new_value)
    
    def _get_simulated_packet_loss(self) -> float:
        """Simula la percentuale di perdita pacchetti."""
        # Normalmente bassa, ma occasionalmente più alta
        if random.random() < 0.1:  # 10% di probabilità di spike
            return random.uniform(1.0, 5.0)  # 1-5% di packet loss
        else:
            return random.uniform(0.0, 0.5)  # 0-0.5% di packet loss normale
    
    def _get_simulated_latency(self) -> float:
        """Simula la latenza di rete in millisecondi."""
        # Base latency plus occasional spikes
        base_latency = random.uniform(5.0, 20.0)  # 5-20ms base
        
        if random.random() < 0.05:  # 5% di probabilità di spike
            spike = random.uniform(50.0, 200.0)  # 50-200ms di spike
            return base_latency + spike
        else:
            return base_latency
    
    def _get_simulated_connection_count(self) -> int:
        """Simula il conteggio delle connessioni attive."""
        # Segue un pattern giornaliero
        hour_of_day = time.localtime().tm_hour
        
        # Più connessioni durante l'orario lavorativo
        if 9 <= hour_of_day <= 17:
            return random.randint(50, 200)
        else:
            return random.randint(10, 50)