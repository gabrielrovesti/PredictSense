import os
import uuid
import logging
import asyncio
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from src.common.messaging import MessageBroker
from src.common.database import Database
from src.common.cache import CacheClient
from src.trainer.models.autoencoder import AnomalyAutoencoder
from src.trainer.models.lstm import TimeSeriesAnomalyDetector
from src.trainer.models.ensemble import AnomalyEnsemble


class DetectorService:
    """
    Servizio di rilevamento anomalie.
    
    Questo servizio carica i modelli ML addestrati e li utilizza per
    rilevare anomalie nei dati in tempo reale.
    """
    
    def __init__(self):
        """Inizializza il servizio detector."""
        self.logger = logging.getLogger(__name__)
        
        # Parametri di configurazione
        self.models_dir = os.getenv("MODELS_DIR", "/app/models")
        self.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.8"))
        self.cache_predictions = os.getenv("CACHE_PREDICTIONS", "true").lower() == "true"
        self.cache_ttl = int(os.getenv("CACHE_TTL", "3600"))  # 1 ora
        
        # Servizi esterni
        self.message_broker = None
        self.database = None
        self.cache = None
        
        # Modelli attivi
        self.models = {}
        self.default_model_id = None
        
        # RPC handlers
        self.rpc_handlers = {
            "detector.predict": self.handle_predict,
            "detector.status": self.handle_status,
            "detector.reload_models": self.handle_reload_models
        }
    
    async def start(self):
        """Avvia il servizio detector."""
        self.logger.info("Starting detector service")
        
        # Connessione al message broker
        self.message_broker = MessageBroker(
            host=os.getenv("RABBITMQ_HOST", "localhost"),
            port=int(os.getenv("RABBITMQ_PORT", "5672")),
            username=os.getenv("RABBITMQ_USER", "guest"),
            password=os.getenv("RABBITMQ_PASSWORD", "guest"),
            vhost=os.getenv("RABBITMQ_VHOST", "/"),
            connection_name="detector-service"
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
        
        # Carica i modelli
        await self.load_models()
        
        # Registra RPC handlers
        for routing_key, handler in self.rpc_handlers.items():
            await self._register_rpc_handler(routing_key, handler)
        
        self.logger.info("Detector service started successfully")
    
    async def stop(self):
        """Ferma il servizio detector."""
        self.logger.info("Stopping detector service")
        
        # Chiusura connessioni
        if self.message_broker:
            await self.message_broker.close()
        
        if self.database:
            await self.database.close()
        
        if self.cache:
            await self.cache.close()
        
        self.logger.info("Detector service stopped")
    
    async def _register_rpc_handler(self, routing_key, handler):
        """Registra un handler per le chiamate RPC."""
        queue_name = f"rpc.{routing_key}"
        
        # Crea la coda
        await self.message_broker.get_queue(
            queue_name=queue_name,
            durable=True
        )
        
        # Crea il binding con l'exchange
        await self.message_broker.get_exchange(
            exchange_name="predictsense",
            exchange_type="topic",
            durable=True
        )
        
        # Registra il callback
        async def rpc_callback(payload, message):
            correlation_id = message.correlation_id
            reply_to = message.reply_to
            
            if not correlation_id or not reply_to:
                self.logger.warning("Received RPC request without correlation_id or reply_to")
                return
            
            try:
                # Gestisci la richiesta
                result = await handler(payload)
                
                # Pubblica la risposta
                await self.message_broker.publish(
                    exchange_name="",  # Default exchange for direct routing to queue
                    routing_key=reply_to,
                    message=result,
                    correlation_id=correlation_id
                )
                
            except Exception as e:
                self.logger.exception(f"Error processing RPC request: {str(e)}")
                
                # Pubblica risposta con errore
                error_response = {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                
                await self.message_broker.publish(
                    exchange_name="",
                    routing_key=reply_to,
                    message=error_response,
                    correlation_id=correlation_id
                )
        
        # Sottoscrivi alla coda
        await self.message_broker.subscribe(
            queue_name=queue_name,
            callback=rpc_callback,
            exchange_name="predictsense",
            routing_key=routing_key,
            auto_ack=False
        )
        
        self.logger.info(f"Registered RPC handler for routing key: {routing_key}")
    
    async def load_models(self):
        """Carica i modelli di ML dal disco."""
        self.logger.info("Loading ML models")
        
        try:
            # Verifica che la directory dei modelli esista
            if not os.path.exists(self.models_dir):
                os.makedirs(self.models_dir, exist_ok=True)
                self.logger.warning(f"Models directory {self.models_dir} created")
                return {}
            
            # Carica i modelli attivi dal database
            query = """
                SELECT model_id, model_path, model_type, is_default
                FROM models
                WHERE status = 'active'
            """
            
            model_records = await self.database.fetch_all(query)
            
            if not model_records:
                self.logger.warning("No active models found in database")
                return {}
            
            # Carica ogni modello
            for record in model_records:
                model_id = record['model_id']
                model_path = record['model_path']
                model_type = record['model_type']
                is_default = record['is_default']
                
                try:
                    # Percorso completo
                    full_path = os.path.join(self.models_dir, model_path)
                    
                    if not os.path.exists(full_path):
                        self.logger.warning(f"Model path {full_path} does not exist")
                        continue
                    
                    # Carica in base al tipo
                    if model_type == 'autoencoder':
                        model = tf.keras.models.load_model(
                            full_path,
                            custom_objects={'AnomalyAutoencoder': AnomalyAutoencoder}
                        )
                    elif model_type == 'lstm':
                        model = tf.keras.models.load_model(
                            full_path,
                            custom_objects={'TimeSeriesAnomalyDetector': TimeSeriesAnomalyDetector}
                        )
                    elif model_type == 'ensemble':
                        model = AnomalyEnsemble.load(full_path)
                    else:
                        self.logger.warning(f"Unsupported model type: {model_type}")
                        continue
                    
                    # Aggiungi ai modelli attivi
                    self.models[model_id] = {
                        'model': model,
                        'type': model_type,
                        'path': model_path
                    }
                    
                    # Imposta come default se specificato
                    if is_default:
                        self.default_model_id = model_id
                    
                    self.logger.info(f"Loaded model {model_id} of type {model_type}")
                    
                except Exception as e:
                    self.logger.exception(f"Error loading model {model_id}: {str(e)}")
            
            # Verifica che sia stato caricato almeno un modello
            if not self.models:
                self.logger.warning("No models could be loaded")
            else:
                # Se nessun modello è marcato come default, usa il primo
                if not self.default_model_id and self.models:
                    self.default_model_id = next(iter(self.models))
                    self.logger.info(f"No default model specified, using {self.default_model_id}")
                
                self.logger.info(f"Loaded {len(self.models)} models")
            
            return self.models
            
        except Exception as e:
            self.logger.exception(f"Error loading models: {str(e)}")
            return {}
    
    async def handle_reload_models(self, payload):
        """Handler per ricaricare i modelli."""
        self.logger.info("Reloading ML models")
        
        try:
            # Pulisci i modelli esistenti
            self.models = {}
            self.default_model_id = None
            
            # Ricarica i modelli
            models = await self.load_models()
            
            return {
                "success": True,
                "models_loaded": len(models),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.exception(f"Error reloading models: {str(e)}")
            
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def handle_status(self, payload):
        """Handler per verificare lo stato del detector."""
        return {
            "status": "running",
            "models_loaded": len(self.models),
            "default_model": self.default_model_id,
            "timestamp": datetime.now().isoformat()
        }
    
    async def handle_predict(self, payload):
        """
        Handler per le richieste di predizione.
        
        Args:
            payload: Dati per la predizione
        
        Returns:
            Risultato della predizione
        """
        self.logger.debug(f"Received prediction request: {json.dumps(payload, default=str)[:100]}...")
        
        try:
            # Verifica che ci siano modelli caricati
            if not self.models:
                raise ValueError("No models loaded")
            
            # Estrai dati dal payload
            source_id = payload.get('source_id')
            features = payload.get('features')
            metadata = payload.get('metadata', {})
            model_id = metadata.get('model_id', self.default_model_id)
            
            # Verifica dati
            if not features:
                raise ValueError("No features provided")
            
            # Se il modello specificato non esiste, usa il default
            if model_id not in self.models:
                self.logger.warning(f"Model {model_id} not found, using default: {self.default_model_id}")
                model_id = self.default_model_id
            
            # Ottieni il modello
            model_info = self.models[model_id]
            model = model_info['model']
            model_type = model_info['type']
            
            # Converti features in numpy array
            if isinstance(features, list):
                features_array = np.array(features)
            else:
                features_array = np.array([features])
            
            # Rileva anomalie in base al tipo di modello
            prediction_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            if model_type == 'autoencoder':
                anomaly_detected, anomaly_score, confidence = self._detect_with_autoencoder(
                    model, features_array
                )
            elif model_type == 'lstm':
                # Per LSTM assumiamo che features sia già nella forma corretta
                anomaly_detected, anomaly_score, confidence = self._detect_with_lstm(
                    model, features_array
                )
            elif model_type == 'ensemble':
                # Per ensemble prepariamo i dati nel formato richiesto
                ensemble_data = self._prepare_ensemble_data(features_array, model_id)
                anomaly_detected, anomaly_score, confidence = self._detect_with_ensemble(
                    model, ensemble_data
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Prepara risposta
            result = {
                "prediction_id": prediction_id,
                "source_id": source_id,
                "anomaly_detected": bool(anomaly_detected),
                "anomaly_score": float(anomaly_score),
                "confidence": float(confidence),
                "model_id": model_id,
                "timestamp": timestamp
            }
            
            # Salva predizione in background
            asyncio.create_task(
                self._save_prediction(result)
            )
            
            # Pubblica evento
            asyncio.create_task(
                self._publish_prediction_event(result)
            )
            
            self.logger.info(
                f"Prediction {prediction_id}: anomaly={anomaly_detected}, "
                f"score={anomaly_score:.4f}, confidence={confidence:.4f}"
            )
            
            return result
            
        except Exception as e:
            self.logger.exception(f"Error processing prediction request: {str(e)}")
            
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _detect_with_autoencoder(
        self, model: AnomalyAutoencoder, features: np.ndarray
    ) -> Tuple[bool, float, float]:
        """
        Rileva anomalie utilizzando un modello autoencoder.
        
        Args:
            model: Modello autoencoder
            features: Features da analizzare
        
        Returns:
            Tupla di (anomaly_detected, anomaly_score, confidence)
        """
        # Verifica che il modello abbia una soglia impostata
        if model.threshold is None:
            # Usa una soglia di default se non impostata
            threshold = 0.5
        else:
            threshold = model.threshold
        
        # Calcola l'errore di ricostruzione
        reconstruction_error = model.compute_reconstruction_error(features).numpy()
        
        # Se è un batch, prendi la media
        if len(reconstruction_error.shape) > 0:
            anomaly_score = float(np.mean(reconstruction_error))
        else:
            anomaly_score = float(reconstruction_error)
        
        # Normalizza lo score in [0, 1]
        normalized_score = min(1.0, max(0.0, anomaly_score / (threshold * 2)))
        
        # Determina se è un'anomalia
        anomaly_detected = anomaly_score > threshold
        
        # Calcola la confidence come distanza dalla soglia
        distance_from_threshold = abs(anomaly_score - threshold) / threshold
        confidence = min(1.0, max(0.5, 0.5 + distance_from_threshold))
        
        return anomaly_detected, normalized_score, confidence
    
    def _detect_with_lstm(
        self, model: TimeSeriesAnomalyDetector, features: np.ndarray
    ) -> Tuple[bool, float, float]:
        """
        Rileva anomalie utilizzando un modello LSTM.
        
        Args:
            model: Modello LSTM
            features: Features da analizzare (ultimi valori noti)
        
        Returns:
            Tupla di (anomaly_detected, anomaly_score, confidence)
        """
        # Ottiene la previsione
        prediction = model.predict(features)
        
        # Calcola l'errore di previsione (MSE)
        # Assume che l'ultimo valore sia il target
        actual = features[:, -1, :]
        mse = np.mean(np.square(actual - prediction))
        
        # Verifica che il modello abbia una soglia impostata
        if model.threshold is None:
            # Usa una soglia di default se non impostata
            threshold = 0.5
        else:
            threshold = model.threshold
        
        # Normalizza lo score in [0, 1]
        anomaly_score = min(1.0, max(0.0, mse / (threshold * 2)))
        
        # Determina se è un'anomalia
        anomaly_detected = mse > threshold
        
        # Calcola la confidence come distanza dalla soglia
        distance_from_threshold = abs(mse - threshold) / threshold
        confidence = min(1.0, max(0.5, 0.5 + distance_from_threshold))
        
        return anomaly_detected, anomaly_score, confidence
    
    def _detect_with_ensemble(
        self, model: AnomalyEnsemble, data: Dict[str, Any]
    ) -> Tuple[bool, float, float]:
        """
        Rileva anomalie utilizzando un modello ensemble.
        
        Args:
            model: Modello ensemble
            data: Dati formattati per il modello ensemble
        
        Returns:
            Tupla di (anomaly_detected, anomaly_score, confidence)
        """
        # Utilizza l'ensemble per la predizione
        anomalies, scores = model.detect_anomalies(data)
        
        # Se è un batch, prendi la media
        if len(scores.shape) > 0:
            anomaly_score = float(np.mean(scores))
            anomaly_detected = bool(np.any(anomalies))
        else:
            anomaly_score = float(scores)
            anomaly_detected = bool(anomalies)
        
        # Normalizza lo score in [0, 1] (assumendo che scores è già normalizzato)
        normalized_score = min(1.0, max(0.0, anomaly_score))
        
        # Calcola la confidence in base alla distribuzione degli scores
        if len(scores.shape) > 0 and len(scores) > 1:
            # Se abbiamo più scores, usa la varianza come misura di confidence
            confidence = 1.0 - min(0.5, float(np.std(scores)))
        else:
            # Altrimenti usa una confidence proporzionale allo score
            confidence = 0.5 + (normalized_score * 0.5 if anomaly_detected else (1 - normalized_score) * 0.5)
        
        return anomaly_detected, normalized_score, confidence
    
    def _prepare_ensemble_data(self, features: np.ndarray, model_id: str) -> Dict[str, Any]:
        """
        Prepara i dati nel formato richiesto dall'ensemble.
        
        Args:
            features: Features originali
            model_id: ID del modello
        
        Returns:
            Dati formattati per l'ensemble
        """
        # Questo è un esempio semplificato, andrebbe personalizzato in base all'implementazione reale
        return {
            f"{model_id}": features
        }
    
    async def _save_prediction(self, prediction: Dict[str, Any]):
        """
        Salva il risultato della predizione nel database.
        
        Args:
            prediction: Risultato della predizione
        """
        try:
            # Inserisci nel database
            query = """
                INSERT INTO predictions (
                    prediction_id, source_id, anomaly_detected,
                    anomaly_score, confidence, model_id, timestamp,
                    created_at
                )
                VALUES (
                    $1, $2, $3, $4, $5, $6, $7, NOW()
                )
            """
            
            await self.database.execute(
                query,
                prediction["prediction_id"],
                prediction["source_id"],
                prediction["anomaly_detected"],
                prediction["anomaly_score"],
                prediction["confidence"],
                prediction["model_id"],
                prediction["timestamp"]
            )
            
            # Salva in cache se abilitato
            if self.cache_predictions and self.cache:
                await self.cache.set(
                    f"prediction:{prediction['prediction_id']}",
                    json.dumps(prediction),
                    expire=self.cache_ttl
                )
            
        except Exception as e:
            self.logger.error(f"Error saving prediction {prediction['prediction_id']}: {str(e)}")
    
    async def _publish_prediction_event(self, prediction: Dict[str, Any]):
        """
        Pubblica un evento per la predizione completata.
        
        Args:
            prediction: Risultato della predizione
        """
        try:
            routing_key = "events.prediction.completed"
            if prediction["anomaly_detected"]:
                routing_key = "events.prediction.anomaly_detected"
            
            await self.message_broker.publish(
                exchange_name="predictsense",
                routing_key=routing_key,
                message=prediction
            )
            
        except Exception as e:
            self.logger.error(
                f"Error publishing event for prediction {prediction['prediction_id']}: {str(e)}"
            )


# Entrypoint per l'esecuzione come servizio
async def main():
    # Configurazione logging
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Crea e avvia il servizio
    detector = DetectorService()
    
    try:
        await detector.start()
        
        # Mantieni il servizio attivo
        while True:
            await asyncio.sleep(60)
            
    except KeyboardInterrupt:
        logging.info("Detector service interrupted")
    except Exception as e:
        logging.exception(f"Error in detector service: {str(e)}")
    finally:
        await detector.stop()


if __name__ == "__main__":
    asyncio.run(main())