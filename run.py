"""
Script per avviare i componenti dell'applicazione PredictSense individualmente.
Utile per lo sviluppo e il testing senza Docker.
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

# Assicura che stiamo eseguendo dalla directory del progetto
project_root = Path(__file__).parent.absolute()
os.chdir(project_root)

# Assicura che src sia nel path di Python
sys.path.insert(0, str(project_root))

def run_api():
    """Avvia il servizio API con Uvicorn."""
    print("Avvio del servizio API...")
    subprocess.run(["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])

def run_dashboard():
    """Avvia la dashboard con Streamlit."""
    print("Avvio della dashboard...")
    subprocess.run(["streamlit", "run", "src/dashboard/app.py"])

def run_detector():
    """Avvia il servizio detector."""
    print("Avvio del servizio detector...")
    from src.detector.detector_service import main
    import asyncio
    asyncio.run(main())

def setup_environment():
    """Configura l'ambiente di sviluppo."""
    print("Configurazione dell'ambiente di sviluppo...")
    
    # Crea directory per i modelli se non esiste
    models_dir = os.path.join(project_root, "data", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Verifica se il file .env esiste
    env_file = os.path.join(project_root, ".env")
    if not os.path.exists(env_file):
        print("File .env non trovato. Creazione di un file .env di esempio...")
        with open(env_file, "w") as f:
            f.write("""# PostgreSQL Configuration
POSTGRES_USER=predictsense
POSTGRES_PASSWORD=predictsensepass
POSTGRES_DB=predictsense
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# RabbitMQ Configuration
RABBITMQ_USER=predictsense
RABBITMQ_PASSWORD=predictsensepass
RABBITMQ_HOST=localhost
RABBITMQ_PORT=5672
RABBITMQ_VHOST=/

# Redis Configuration
REDIS_PASSWORD=predictsensepass
REDIS_HOST=localhost
REDIS_PORT=6379

# API Configuration
API_KEY=predictsense_api_key
ENABLE_CORS=true

# ML Configuration
MODELS_DIR=./data/models
CONFIDENCE_THRESHOLD=0.8
CACHE_PREDICTIONS=true
CACHE_TTL=3600

# Logging
LOG_LEVEL=INFO
""")
    
    print("Ambiente configurato con successo!")

def main():
    parser = argparse.ArgumentParser(description="PredictSense Runner")
    parser.add_argument(
        "service", 
        choices=["api", "dashboard", "detector", "setup"],
        help="Servizio da avviare o azione da eseguire"
    )
    
    args = parser.parse_args()
    
    if args.service == "api":
        run_api()
    elif args.service == "dashboard":
        run_dashboard()
    elif args.service == "detector":
        run_detector()
    elif args.service == "setup":
        setup_environment()

if __name__ == "__main__":
    main()