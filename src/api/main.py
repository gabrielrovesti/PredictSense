# src/api/main.py
import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from src.common.messaging import MessageBroker
from src.common.messaging_helpers import RPCClient
from src.common.database import Database
from src.common.cache import CacheClient
from src.api.models.schemas import (
    PredictionRequest,
    PredictionResponse,
    AnomalyResponse,
    ModelInfo,
    MetricsResponse,
    ErrorResponse
)
from src.api.routers import predictions, models, metrics
from src.api.dependencies import get_message_broker, get_db, get_cache
from src.api.middleware import RequestLoggerMiddleware, PrometheusMiddleware

# Configurazione logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Inizializzazione app
app = FastAPI(
    title="PredictSense API",
    description="API per il sistema di rilevamento anomalie",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Configurazione CORS
if os.getenv("ENABLE_CORS", "true").lower() == "true":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In produzione, specificare i domini consentiti
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Middleware
app.add_middleware(RequestLoggerMiddleware)
app.add_middleware(PrometheusMiddleware)

# Sicurezza API
API_KEY = os.getenv("API_KEY", "predictsense_api_key")
api_key_header = APIKeyHeader(name="X-API-Key")


async def verify_api_key(api_key: str = Depends(api_key_header)):
    """Verifica l'API key."""
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return api_key


# Inclusione router
app.include_router(
    predictions.router,
    prefix="/predictions",
    tags=["predictions"],
    dependencies=[Depends(verify_api_key)]
)
app.include_router(
    models.router,
    prefix="/models",
    tags=["models"],
    dependencies=[Depends(verify_api_key)]
)
app.include_router(
    metrics.router,
    prefix="/metrics",
    tags=["metrics"],
    dependencies=[Depends(verify_api_key)]
)


@app.on_event("startup")
async def startup_event():
    """Inizializzazione servizi all'avvio."""
    logger.info("Starting API service")
    
    # Connessione al message broker
    app.state.message_broker = MessageBroker(
        host=os.getenv("RABBITMQ_HOST", "localhost"),
        port=int(os.getenv("RABBITMQ_PORT", "5672")),
        username=os.getenv("RABBITMQ_USER", "guest"),
        password=os.getenv("RABBITMQ_PASSWORD", "guest"),
        vhost=os.getenv("RABBITMQ_VHOST", "/"),
        connection_name="api-service"
    )
    await app.state.message_broker.connect()
    
    # Inizializza RPC client
    app.state.rpc_client = RPCClient(app.state.message_broker)
    await app.state.rpc_client.initialize()
    
    # Connessione al database
    app.state.db = Database(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        username=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        database=os.getenv("POSTGRES_DB", "predictsense"),
        max_connections=10
    )
    await app.state.db.connect()
    
    # Connessione al cache
    app.state.cache = CacheClient(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        password=os.getenv("REDIS_PASSWORD", ""),
        db=0
    )
    await app.state.cache.connect()
    
    logger.info("API service started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Pulizia risorse alla chiusura."""
    logger.info("Shutting down API service")
    
    # Chiusura connessioni
    await app.state.message_broker.close()
    await app.state.db.close()
    await app.state.cache.close()
    
    logger.info("API service shutdown complete")


@app.get("/", tags=["status"])
async def root():
    """Root endpoint che restituisce lo stato del servizio."""
    return {
        "service": "PredictSense API",
        "status": "running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health", tags=["status"])
async def health_check():
    """Endpoint per health check."""
    # Verifica stato connessioni
    checks = {
        "database": await app.state.db.is_connected(),
        "message_broker": app.state.message_broker.connection is not None 
                         and not app.state.message_broker.connection.is_closed,
        "cache": await app.state.cache.is_connected()
    }
    
    all_healthy = all(checks.values())
    
    if not all_healthy:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "checks": checks,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    return {
        "status": "healthy",
        "checks": checks,
        "timestamp": datetime.now().isoformat()
    }


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handler per eccezioni generiche."""
    logger.exception(f"Unhandled exception: {str(exc)}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            message=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )


