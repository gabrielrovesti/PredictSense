# Esempio di uno dei router

import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Path, BackgroundTasks
from fastapi.responses import JSONResponse

from src.api.models.schemas import (
    PredictionRequest,
    PredictionResponse,
    AnomalyResponse
)
from src.api.dependencies import get_message_broker, get_db, get_cache, get_rpc_client
from src.common.messaging import MessageBroker
from src.common.messaging_helpers import RPCClient
from src.common.database import Database
from src.common.cache import CacheClient

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    message_broker: MessageBroker = Depends(get_message_broker),
    rpc_client: RPCClient = Depends(get_rpc_client),
    cache: CacheClient = Depends(get_cache)
):
    """
    Predice anomalie nei dati forniti.
    
    I dati vengono inviati al servizio di rilevamento per l'analisi.
    I risultati vengono restituiti immediatamente e pubblicati anche come evento.
    """
    logger.info(f"Received prediction request: {request.model_dump(exclude={'features'})}")
    
    start_time = datetime.now()
    
    try:
        # Invio richiesta al servizio detector via RPC
        prediction_data = {
            "source_id": request.source_id,
            "features": request.features,
            "metadata": request.metadata,
            "timestamp": datetime.now().isoformat()
        }
        
        # Chiamata RPC al servizio detector
        result = await rpc_client.call(
            exchange_name="predictsense",
            routing_key="detector.predict",
            payload=prediction_data,
            timeout=10.0
        )
        
        if not result or "error" in result:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {result.get('error', 'Unknown error')}"
            )
        
        # Calcola tempo di elaborazione
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Logga risultato su database in background
        background_tasks.add_task(
            _log_prediction_result,
            request.source_id,
            result,
            processing_time,
            message_broker
        )
        
        # Costruisci risposta
        response = PredictionResponse(
            prediction_id=result["prediction_id"],
            anomaly_detected=result["anomaly_detected"],
            anomaly_score=result["anomaly_score"],
            confidence=result["confidence"],
            processing_time_ms=processing_time,
            model_id=result["model_id"],
            timestamp=result["timestamp"]
        )
        
        return response
        
    except asyncio.TimeoutError:
        logger.error("Prediction request timed out")
        raise HTTPException(
            status_code=504,
            detail="Prediction request timed out"
        )
        
    except Exception as e:
        logger.exception(f"Error during prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.get("/{prediction_id}", response_model=PredictionResponse)
async def get_prediction(
    prediction_id: str = Path(..., description="ID della predizione"),
    db: Database = Depends(get_db),
    cache: CacheClient = Depends(get_cache)
):
    """
    Recupera i dettagli di una predizione specifica.
    
    Cerca prima nella cache, poi nel database se non trovata.
    """
    logger.info(f"Retrieving prediction with ID: {prediction_id}")
    
    # Cerca in cache
    cached_prediction = await cache.get(f"prediction:{prediction_id}")
    if cached_prediction:
        return PredictionResponse(**json.loads(cached_prediction))
    
    # Cerca nel database
    query = """
        SELECT 
            prediction_id, 
            source_id,
            anomaly_detected,
            anomaly_score,
            confidence,
            processing_time_ms,
            model_id,
            timestamp
        FROM 
            predictions
        WHERE 
            prediction_id = $1
    """
    
    result = await db.fetch_one(query, prediction_id)
    
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Prediction with ID {prediction_id} not found"
        )
    
    # Costruisci risposta
    prediction = dict(result)
    
    # Salva in cache per riutilizzo
    await cache.set(
        f"prediction:{prediction_id}",
        json.dumps(prediction),
        expire=3600  # 1 ora
    )
    
    return PredictionResponse(**prediction)


@router.get("/", response_model=List[PredictionResponse])
async def list_predictions(
    source_id: Optional[str] = Query(None, description="Filtra per ID sorgente"),
    model_id: Optional[str] = Query(None, description="Filtra per ID modello"),
    anomaly_only: bool = Query(False, description="Filtra solo anomalie"),
    limit: int = Query(20, ge=1, le=100, description="Numero massimo di risultati"),
    offset: int = Query(0, ge=0, description="Offset per paginazione"),
    db: Database = Depends(get_db)
):
    """
    Elenca le predizioni effettuate.
    
    Supporta filtri e paginazione.
    """
    logger.info(f"Listing predictions with filters: source_id={source_id}, model_id={model_id}, anomaly_only={anomaly_only}")
    
    # Costruisci query
    query_parts = ["SELECT * FROM predictions"]
    params = []
    
    # Aggiungi filtri
    filters = []
    
    if source_id:
        filters.append(f"source_id = ${len(params) + 1}")
        params.append(source_id)
    
    if model_id:
        filters.append(f"model_id = ${len(params) + 1}")
        params.append(model_id)
    
    if anomaly_only:
        filters.append("anomaly_detected = true")
    
    if filters:
        query_parts.append("WHERE " + " AND ".join(filters))
    
    # Aggiungi ordinamento e paginazione
    query_parts.append("ORDER BY timestamp DESC")
    query_parts.append(f"LIMIT ${len(params) + 1}")
    params.append(limit)
    
    query_parts.append(f"OFFSET ${len(params) + 1}")
    params.append(offset)
    
    # Esegui query
    results = await db.fetch_all(" ".join(query_parts), *params)
    
    # Formatta risultati
    predictions = [PredictionResponse(**dict(row)) for row in results]
    
    return predictions


async def _log_prediction_result(
    source_id: str,
    result: Dict[str, Any],
    processing_time: float,
    message_broker: MessageBroker
):
    """
    Salva il risultato della predizione e pubblica un evento.
    
    Questa funzione viene eseguita in background.
    """
    # Pubblica evento
    await message_broker.publish(
        exchange_name="predictsense",
        routing_key="events.prediction.completed",
        message={
            "prediction_id": result["prediction_id"],
            "source_id": source_id,
            "anomaly_detected": result["anomaly_detected"],
            "anomaly_score": result["anomaly_score"],
            "timestamp": result["timestamp"],
            "processing_time_ms": processing_time
        }
    )