"""
Fonte dati HTTP per il collector.
Raccoglie dati da fonti HTTP come API, logs, ecc.
"""

import os
import logging
import time
import json
import random
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional


class HTTPDataSource:
    """
    Fonte dati per sorgenti HTTP/API.
    
    Raccoglie dati da API REST, log server, ecc.
    """
    
    def __init__(self, source_id: str, config: Dict[str, Any]):
        """
        Inizializza la fonte dati HTTP.
        
        Args:
            source_id: Identificativo della fonte dati
            config: Configurazione della fonte dati
        """
        self.source_id = source_id
        self.config = config if isinstance(config, dict) else json.loads(config)
        self.logger = logging.getLogger(__name__)
        
        # Parametri di configurazione
        self.url = self.config.get("url", "")
        self.method = self.config.get("method", "GET")
        self.headers = self.config.get("headers", {})
        self.timeout = self.config.get("timeout", 30)
        self.pattern = self.config.get("pattern", "ERROR|WARN")
        
        # Solo per simulazione
        self.log_path = self.config.get("log_path", "/var/log/application")
        
        # Configurazione del polling
        self.collection_interval = self.config.get("collection_interval", 300)
    
    async def collect(self) -> Dict[str, Any]:
        """
        Raccoglie i dati dalla fonte HTTP.
        
        In un'implementazione reale, questo metodo effettuerebbe 
        una chiamata HTTP. Per dimostrazione, genera dati simulati.
        
        Returns:
            Dizionario di dati raccolti
        """
        try:
            if self.url:
                return await self._collect_from_api()
            else:
                return await self._collect_from_logs()
            
        except Exception as e:
            self.logger.error(f"Error collecting HTTP data: {str(e)}")
            return {}
    
    async def _collect_from_api(self) -> Dict[str, Any]:
        """
        Raccoglie dati da un'API HTTP.
        
        In un'implementazione reale, effettuerebbe una chiamata HTTP.
        Per dimostrazione, genera dati simulati.
        
        Returns:
            Dizionario di dati raccolti
        """
        # Simulazione raccolta dati
        # In un'implementazione reale, usare aiohttp per effettuare chiamate HTTP
        
        # Simula l'aspetto di una response HTTP
        result = {
            "timestamp": time.time(),
            "status": 200,
            "response_time": random.uniform(0.05, 0.5),  # 50-500ms
            "data": self._get_simulated_api_data()
        }
        
        # Simulare occasionalmente errori HTTP
        if random.random() < 0.05:  # 5% di probabilità di errore
            result["status"] = random.choice([400, 401, 403, 404, 500, 502, 503])
            result["data"] = {"error": f"Simulated HTTP error {result['status']}"}
        
        return result
    
    async def _collect_from_logs(self) -> Dict[str, Any]:
        """
        Raccoglie dati da log applicativi.
        
        In un'implementazione reale, leggerebbe file di log.
        Per dimostrazione, genera dati simulati.
        
        Returns:
            Dizionario di dati raccolti
        """
        # Simulazione raccolta dati
        # In un'implementazione reale, leggere i file di log
        
        # Genera log entries simulati
        num_entries = random.randint(5, 20)
        entries = []
        
        for _ in range(num_entries):
            entry = self._get_simulated_log_entry()
            entries.append(entry)
        
        return {
            "timestamp": time.time(),
            "log_path": self.log_path,
            "pattern": self.pattern,
            "entries": entries,
            "count": len(entries)
        }
    
    def _get_simulated_api_data(self) -> Dict[str, Any]:
        """Genera dati API simulati."""
        # Simuliamo dati da un'API di monitoraggio applicativo
        
        endpoints = ["login", "home", "search", "product", "checkout", "payment", "profile"]
        
        # Genera metriche per alcuni endpoint casuali
        result = {"metrics": []}
        
        for _ in range(random.randint(3, 7)):
            endpoint = random.choice(endpoints)
            
            metric = {
                "endpoint": f"/{endpoint}",
                "requests": random.randint(100, 5000),
                "average_response_time": random.uniform(10, 500),
                "error_rate": random.uniform(0, 3),
                "status": {
                    "200": random.randint(90, 100),
                    "400": random.randint(0, 5),
                    "500": random.randint(0, 3)
                }
            }
            
            result["metrics"].append(metric)
        
        return result
    
    def _get_simulated_log_entry(self) -> Dict[str, Any]:
        """Genera una entry di log simulata."""
        log_levels = ["DEBUG", "INFO", "WARN", "ERROR", "FATAL"]
        
        # Distribuisci i livelli in modo realistico
        level_weights = [50, 30, 10, 8, 2]  # percentuali
        level = random.choices(log_levels, weights=level_weights, k=1)[0]
        
        # Simula un timestamp negli ultimi minuti
        current_time = time.time()
        log_time = current_time - random.uniform(0, self.collection_interval)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(log_time))
        
        # Genera un messaggio di log appropriato
        if level == "DEBUG":
            message = random.choice([
                "Detailed debugging information",
                "Variable state: key=value",
                "Function execution completed in 25ms",
                "Cache hit ratio: 0.87"
            ])
        elif level == "INFO":
            message = random.choice([
                "Application started successfully",
                "User logged in: user123",
                "Database connection established",
                "Scheduled task completed"
            ])
        elif level == "WARN":
            message = random.choice([
                "Slow database query detected (>200ms)",
                "Configuration parameter missing, using default",
                "Deprecated API call used",
                "Multiple failed login attempts for user"
            ])
        elif level == "ERROR":
            message = random.choice([
                "Failed to connect to database: Connection timeout",
                "API request failed with status 500",
                "Unable to process payment: Invalid card",
                "File not found: /data/config.json"
            ])
        else:  # FATAL
            message = random.choice([
                "Application crashed: Out of memory",
                "Fatal database corruption detected",
                "Unrecoverable system error",
                "Critical security breach detected"
            ])
        
        return {
            "timestamp": timestamp,
            "level": level,
            "message": message,
            "service": random.choice(["api", "auth", "database", "frontend", "payment"]),
            "thread": f"thread-{random.randint(1, 10)}"
        }