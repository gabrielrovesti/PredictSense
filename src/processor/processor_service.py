"""
Servizio di processing dati per PredictSense.
Responsabile del preprocessing dei dati raccolti prima del rilevamento anomalie.
"""

import os
import logging
import asyncio
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

from src.common.messaging import MessageBroker
from src.common.database import Database
from src.common.cache import CacheClient


class ProcessorService:
    """
    Servizio per il preprocessing dei dati.
    
    Responsabile di:
    - Applicare trasformazioni ai dati raccolti
    - Normalizzare e pulire i dati
    - Estrarre features per il rilevamento anomalie
    - Pubblicare dati elaborati per il rilevatore
    """
    
    def __init__(self):
        """Inizializza il servizio processor."""
        self.logger = logging.getLogger(__name__)
        
        # Servizi esterni
        self.message_broker = None
        self.database = None
        self.cache = None
        
        # Pipeline di preprocessing per tipo di fonte
        self.pipelines = {}
        
        # Flag di controllo
        self.is_running = False
    
    async def start(self):
        """Avvia il servizio processor."""
        self.logger.info("Starting processor service")
        
        # Connessione al message broker
        self.message_broker = MessageBroker(
            host=os.getenv("RABBITMQ_HOST", "localhost"),
            port=int(os.getenv("RABBITMQ_PORT", "5672")),
            username=os.getenv("RABBITMQ_USER", "guest"),
            password=os.getenv("RABBITMQ_PASSWORD", "guest"),
            vhost=os.getenv("RABBITMQ_VHOST", "/"),
            connection_name="processor-service"
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
        
        # Inizializza le pipeline di preprocessing
        await self._initialize_pipelines()
        
        # Sottoscrivi al topic per dati raccolti
        queue_name = "processor.data_collected"
        
        await self.message_broker.subscribe(
            queue_name=queue_name,
            callback=self._handle_collected_data,
            exchange_name="predictsense",
            routing_key="data.collected",
            auto_ack=False
        )
        
        self.is_running = True
        self.logger.info("Processor service started successfully")
    
    async def stop(self):
        """Ferma il servizio processor."""
        self.logger.info("Stopping processor service")
        
        self.is_running = False
        
        # Chiusura connessioni
        if self.message_broker:
            await self.message_broker.close()
        
        if self.database:
            await self.database.close()
        
        if self.cache:
            await self.cache.close()
        
        self.logger.info("Processor service stopped")
    
    async def _initialize_pipelines(self):
        """Inizializza le pipeline di preprocessing."""
        try:
            # Ottieni informazioni sulle data sources
            query = """
                SELECT source_id, name, source_type, config
                FROM data_sources
                WHERE enabled = TRUE
            """
            
            sources = await self.database.fetch_all(query)
            
            # Crea pipeline per ogni source type
            for source in sources:
                source_id = source['source_id']
                source_type = source['source_type']
                config = source['config']
                
                if source_type not in self.pipelines:
                    # Crea la pipeline in base al tipo
                    if source_type == 'network':
                        from src.processor.feature_engineering.network_pipeline import NetworkPipeline
                        self.pipelines[source_type] = NetworkPipeline()
                    
                    elif source_type == 'system':
                        from src.processor.feature_engineering.system_pipeline import SystemPipeline
                        self.pipelines[source_type] = SystemPipeline()
                    
                    elif source_type == 'logs':
                        from src.processor.feature_engineering.logs_pipeline import LogsPipeline
                        self.pipelines[source_type] = LogsPipeline()
                    
                    else:
                        from src.processor.feature_engineering.default_pipeline import DefaultPipeline
                        self.pipelines[source_type] = DefaultPipeline()
            
            self.logger.info(f"Initialized {len(self.pipelines)} preprocessing pipelines")
            
        except Exception as e:
            self.logger.exception(f"Error initializing pipelines: {str(e)}")
    
    async def _handle_collected_data(self, payload, message):
        """
        Gestisce i dati raccolti dal collector.
        
        Args:
            payload: Dati raccolti
            message: Oggetto messaggio completo
        """
        try:
            # Estrai informazioni dal payload
            source_id = payload.get("source_id")
            data = payload.get("data")
            timestamp = payload.get("timestamp")
            
            if not source_id or not data:
                self.logger.warning("Received invalid data: missing source_id or data")
                await message.reject()
                return
            
            # Ottieni il tipo di source
            source_type = await self._get_source_type(source_id)
            
            if not source_type:
                self.logger.warning(f"Unknown source type for {source_id}")
                await message.reject()
                return
            
            # Applica la pipeline di preprocessing
            if source_type in self.pipelines:
                pipeline = self.pipelines[source_type]
                processed_data = await pipeline.process(data)
                
                # Pubblica i dati processati
                await self._publish_processed_data(source_id, processed_data, timestamp)
                
                # Acknowledge il messaggio
                await message.ack()
                
                self.logger.debug(f"Processed data from source {source_id}")
            else:
                self.logger.warning(f"No pipeline available for source type {source_type}")
                await message.reject()
            
        except Exception as e:
            self.logger.error(f"Error processing collected data: {str(e)}")
            # Reject il messaggio in caso di errore
            await message.reject(requeue=True)
    
    async def _get_source_type(self, source_id: str) -> Optional[str]:
        """
        Ottiene il tipo di fonte dati.
        
        Args:
            source_id: ID della fonte dati
        
        Returns:
            Tipo della fonte dati, None se non trovata
        """
        try:
            # Prima cerca in cache
            cache_key = f"source_type:{source_id}"
            cached_type = await self.cache.get(cache_key)
            
            if cached_type:
                return cached_type
            
            # Altrimenti cerca nel database
            query = "SELECT source_type FROM data_sources WHERE source_id = $1"
            source_type = await self.database.fetch_val(query, source_id)
            
            if source_type:
                # Salva in cache per uso futuro
                await self.cache.set(cache_key, source_type, expire=3600)
            
            return source_type
            
        except Exception as e:
            self.logger.error(f"Error getting source type for {source_id}: {str(e)}")
            return None
    
    async def _publish_processed_data(self, source_id: str, processed_data: Dict[str, Any], timestamp: str):
        """
        Pubblica i dati processati.
        
        Args:
            source_id: ID della fonte dati
            processed_data: Dati elaborati
            timestamp: Timestamp di raccolta
        """
        try:
            # Crea messaggio
            message = {
                "source_id": source_id,
                "features": processed_data,
                "timestamp": timestamp,
                "processor_id": os.getenv("PROCESSOR_ID", "default")
            }
            
            # Pubblica sul message broker
            await self.message_broker.publish(
                exchange_name="predictsense",
                routing_key="data.processed",
                message=message
            )
            
        except Exception as e:
            self.logger.error(f"Error publishing processed data: {str(e)}")