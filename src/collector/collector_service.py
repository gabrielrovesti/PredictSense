"""
Servizio di raccolta dati per PredictSense.
Responsabile dell'acquisizione di dati da diverse fonti e del loro inoltro al sistema.
"""

import os
import logging
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

from src.common.messaging import MessageBroker
from src.common.database import Database
from src.common.cache import CacheClient


class CollectorService:
    """
    Servizio per la raccolta di dati da diverse fonti.
    
    Supporta fonti di dati di rete, sistema e applicazioni.
    I dati raccolti vengono pubblicati sul message broker per essere
    processati dal servizio di pre-elaborazione.
    """
    
    def __init__(self):
        """Inizializza il servizio collector."""
        self.logger = logging.getLogger(__name__)
        
        # Parametri di configurazione
        self.collection_interval = int(os.getenv("COLLECTOR_POLLING_INTERVAL", "60"))
        self.data_sources = {}
        
        # Servizi esterni
        self.message_broker = None
        self.database = None
        self.cache = None
        
        # Flag di controllo
        self.is_running = False
    
    async def start(self):
        """Avvia il servizio collector."""
        self.logger.info("Starting collector service")
        
        # Connessione al message broker
        self.message_broker = MessageBroker(
            host=os.getenv("RABBITMQ_HOST", "localhost"),
            port=int(os.getenv("RABBITMQ_PORT", "5672")),
            username=os.getenv("RABBITMQ_USER", "guest"),
            password=os.getenv("RABBITMQ_PASSWORD", "guest"),
            vhost=os.getenv("RABBITMQ_VHOST", "/"),
            connection_name="collector-service"
        )
        await self.message_broker.connect()
        
        # Connessione al database
        self.database = Database(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            username=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres"),
            database=os.getenv("POSTGRES_DB", "predictsense"),
            max_connections=5
        )
        await self.database.connect()
        
        # Connessione alla cache
        self.cache = CacheClient(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD", ""),
            db=0
        )
        await self.cache.connect()
        
        # Carica le fonti dati
        await self._load_data_sources()
        
        # Avvia il loop di raccolta
        self.is_running = True
        asyncio.create_task(self._collection_loop())
        
        self.logger.info("Collector service started successfully")
    
    async def stop(self):
        """Ferma il servizio collector."""
        self.logger.info("Stopping collector service")
        
        self.is_running = False
        
        # Chiusura connessioni
        if self.message_broker:
            await self.message_broker.close()
        
        if self.database:
            await self.database.close()
        
        if self.cache:
            await self.cache.close()
        
        self.logger.info("Collector service stopped")
    
    async def _load_data_sources(self):
        """Carica le fonti dati configurate dal database."""
        try:
            query = """
                SELECT source_id, name, source_type, config, enabled
                FROM data_sources
                WHERE enabled = TRUE
            """
            
            results = await self.database.fetch_all(query)
            
            for row in results:
                source_id = row['source_id']
                source_type = row['source_type']
                config = row['config']
                
                # Crea la sorgente dati in base al tipo
                if source_type == 'network':
                    from src.collector.data_sources.network_source import NetworkDataSource
                    self.data_sources[source_id] = NetworkDataSource(source_id, config)
                
                elif source_type == 'system':
                    from src.collector.data_sources.system_source import SystemDataSource
                    self.data_sources[source_id] = SystemDataSource(source_id, config)
                
                elif source_type == 'logs':
                    from src.collector.data_sources.http_source import HTTPDataSource
                    self.data_sources[source_id] = HTTPDataSource(source_id, config)
                
                else:
                    self.logger.warning(f"Unknown data source type: {source_type}")
            
            self.logger.info(f"Loaded {len(self.data_sources)} data sources")
            
        except Exception as e:
            self.logger.exception(f"Error loading data sources: {str(e)}")
    
    async def _collection_loop(self):
        """Loop principale per la raccolta dati."""
        while self.is_running:
            try:
                collection_time = datetime.now()
                
                # Raccogli dati da tutte le fonti
                for source_id, source in self.data_sources.items():
                    try:
                        data = await source.collect()
                        
                        if data:
                            await self._process_collected_data(source_id, data, collection_time)
                    
                    except Exception as e:
                        self.logger.error(f"Error collecting from source {source_id}: {str(e)}")
                
                # Attendi il prossimo intervallo
                await asyncio.sleep(self.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception(f"Error in collection loop: {str(e)}")
                await asyncio.sleep(10)  # Attesa più breve in caso di errore
    
    async def _process_collected_data(self, source_id, data, timestamp):
        """
        Elabora i dati raccolti.
        
        Args:
            source_id: ID della fonte dati
            data: Dati raccolti
            timestamp: Timestamp della raccolta
        """
        try:
            # Crea messaggio
            message = {
                "source_id": source_id,
                "data": data,
                "timestamp": timestamp.isoformat(),
                "collector_id": os.getenv("COLLECTOR_ID", "default")
            }
            
            # Pubblica sul message broker
            await self.message_broker.publish(
                exchange_name="predictsense",
                routing_key="data.collected",
                message=message
            )
            
            # Salva nel database
            await self._store_raw_data(source_id, data, timestamp)
            
            self.logger.debug(f"Processed data from source {source_id}")
            
        except Exception as e:
            self.logger.error(f"Error processing data from source {source_id}: {str(e)}")
    
    async def _store_raw_data(self, source_id, data, timestamp):
        """
        Salva i dati grezzi nel database.
        
        Args:
            source_id: ID della fonte dati
            data: Dati raccolti
            timestamp: Timestamp della raccolta
        """
        try:
            query = """
                INSERT INTO raw_data (source_id, data, timestamp)
                VALUES ($1, $2, $3)
                RETURNING data_id
            """
            
            data_id = await self.database.fetch_val(
                query,
                source_id,
                json.dumps(data),
                timestamp
            )
            
            return data_id
            
        except Exception as e:
            self.logger.error(f"Error storing raw data for source {source_id}: {str(e)}")
            return None
