"""
Fonte dati di sistema per il collector.
Raccoglie metriche di sistema come CPU, memoria, disco, ecc.
"""

import os
import logging
import time
import json
import random
from typing import Dict, List, Any, Optional


class SystemDataSource:
    """
    Fonte dati per metriche di sistema.
    
    Raccoglie metriche come:
    - Utilizzo CPU
    - Utilizzo memoria
    - Utilizzo disco
    - I/O wait
    - Load average
    """
    
    def __init__(self, source_id: str, config: Dict[str, Any]):
        """
        Inizializza la fonte dati di sistema.
        
        Args:
            source_id: Identificativo della fonte dati
            config: Configurazione della fonte dati
        """
        self.source_id = source_id
        self.config = config if isinstance(config, dict) else json.loads(config)
        self.logger = logging.getLogger(__name__)
        
        # Configura metriche da raccogliere
        self.metrics = self.config.get("metrics", [
            "cpu_usage", "memory_usage", "disk_usage", "io_wait", "load_average"
        ])
        
        # Configurazione del polling
        self.collection_interval = self.config.get("collection_interval", 30)
    
    async def collect(self) -> Dict[str, Any]:
        """
        Raccoglie le metriche di sistema.
        
        In un'implementazione reale, questo metodo raccoglierebbe 
        effettivamente i dati dal sistema. Per dimostrazione, genera dati simulati.
        
        Returns:
            Dizionario di metriche raccolte
        """
        try:
            # Simulazione raccolta dati
            # In un'implementazione reale, usare librerie come psutil o subprocess
            # per raccogliere effettivamente i dati di sistema
            
            result = {
                "timestamp": time.time()
            }
            
            # Raccoglie metriche richieste
            if "cpu_usage" in self.metrics:
                result["cpu_usage"] = self._get_simulated_cpu_usage()
            
            if "memory_usage" in self.metrics:
                result["memory_usage"] = self._get_simulated_memory_usage()
            
            if "disk_usage" in self.metrics:
                result["disk_usage"] = self._get_simulated_disk_usage()
            
            if "io_wait" in self.metrics:
                result["io_wait"] = self._get_simulated_io_wait()
            
            if "load_average" in self.metrics:
                result["load_average"] = self._get_simulated_load_average()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {str(e)}")
            return {}
    
    def _get_simulated_cpu_usage(self) -> float:
        """Simula l'utilizzo CPU in percentuale."""
        # Normalmente tra 10-40%, occasionalmente più alto
        if random.random() < 0.05:  # 5% di probabilità di carico alto
            return random.uniform(70.0, 95.0)  # 70-95% usage (high)
        else:
            return random.uniform(10.0, 40.0)  # 10-40% usage (normal)
    
    def _get_simulated_memory_usage(self) -> float:
        """Simula l'utilizzo memoria in percentuale."""
        # Memoria più stabile, tipicamente 40-60%
        base_usage = random.uniform(40.0, 60.0)
        
        # Occasionalmente picchi di utilizzo
        if random.random() < 0.1:  # 10% di probabilità di consumo elevato
            spike = random.uniform(10.0, 20.0)
            return min(base_usage + spike, 95.0)  # Cap al 95%
        else:
            return base_usage
    
    def _get_simulated_disk_usage(self) -> float:
        """Simula l'utilizzo disco in percentuale."""
        # Disco solitamente tra 50-70% pieno, con lenta crescita nel tempo
        day_of_month = time.localtime().tm_mday
        # Simulare crescita lenta durante il mese
        growth_factor = day_of_month / 30.0 * 10.0  # 0-10% di crescita nel mese
        
        return random.uniform(50.0, 70.0) + growth_factor
    
    def _get_simulated_io_wait(self) -> float:
        """Simula il tempo di attesa I/O in percentuale."""
        # Normalmente basso, occasionalmente più alto
        if random.random() < 0.1:  # 10% di probabilità di I/O elevato
            return random.uniform(10.0, 30.0)  # 10-30% io wait (high)
        else:
            return random.uniform(0.1, 5.0)  # 0.1-5% io wait (normal)
    
    def _get_simulated_load_average(self) -> List[float]:
        """Simula il load average per 1, 5 e 15 minuti."""
        # Base load dipende dall'ora del giorno
        hour_of_day = time.localtime().tm_hour
        
        # Più carico durante l'orario lavorativo
        if 9 <= hour_of_day <= 17:
            base_load = random.uniform(1.0, 3.0)
        else:
            base_load = random.uniform(0.2, 1.0)
        
        # Load average per 1, 5 e 15 minuti (decrescente)
        load_1 = base_load * random.uniform(0.9, 1.1)
        load_5 = load_1 * random.uniform(0.8, 1.0)  # Leggermente più basso di load_1
        load_15 = load_5 * random.uniform(0.7, 0.95)  # Ancora più basso
        
        return [load_1, load_5, load_15]


