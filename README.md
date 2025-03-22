# PredictSense: Sistema di Anomaly Detection Distribuito

Un sistema completo di machine learning per il rilevamento di anomalie in tempo reale, ottimizzato per ambienti Windows e basato su architetture distribuite.

## 🔍 Panoramica

PredictSense è un sistema end-to-end che implementa una pipeline di machine learning per il rilevamento di anomalie in dati di rete e sistemi. Il progetto dimostra l'integrazione di diverse tecnologie moderne in una soluzione coerente, funzionale ed espandibile.

![Architettura del Sistema](docs/images/architecture_diagram.png)

## 🛠️ Tecnologie Integrate

- **TensorFlow 2.x**: Framework ML per modelli di rilevamento anomalie
- **FastAPI**: Backend API ad alte prestazioni
- **RabbitMQ**: Messaging asincrono tra componenti
- **Docker**: Containerizzazione dei microservizi
- **PostgreSQL**: Storage persistente per i dati
- **Redis**: Caching e gestione sessioni
- **MLflow**: Tracking degli esperimenti ML
- **Grafana/Prometheus**: Monitoraggio e alerting
- **GitHub Actions**: Pipeline CI/CD automatizzata

## 🚀 Caratteristiche Principali

- Architettura a microservizi completamente distribuita
- Preprocessing automatico dei dati con pipeline ETL
- Training incrementale dei modelli ML
- Rilevamento anomalie in tempo reale
- API RESTful per integrazione con altri sistemi
- Dashboard per visualizzazione risultati e metriche
- Deployment semplificato con Docker Compose

## 📋 Prerequisiti

- Windows 10/11 (64-bit)
- Docker Desktop per Windows
- Python 3.9+
- NVIDIA GPU (opzionale, per accelerazione)
- 8GB+ RAM

## 🔧 Installazione

1. **Clona il repository**
   ```
   git clone https://github.com/yourusername/predictsense.git
   cd predictsense
   ```

2. **Configurazione dell'ambiente**
   ```
   # Crea ambiente virtuale
   python -m venv venv
   venv\Scripts\activate
   
   # Installa dipendenze
   pip install -r requirements.txt
   ```

3. **Avvia i servizi con Docker Compose**
   ```
   docker-compose up -d
   ```

4. **Verifica l'installazione**
   ```
   python scripts/verify_setup.py
   ```

## 🧠 Componenti ML

### Pipeline di preprocessing

Il sistema implementa una pipeline di preprocessing utilizzando TensorFlow Data per gestire efficientemente grandi volumi di dati:

```python
def create_preprocessing_pipeline(data_config):
    """Crea una pipeline di preprocessing TensorFlow Data."""
    pipeline = (
        tf.data.Dataset.from_generator(
            data_generator,
            output_signature=data_config.signature
        )
        .batch(data_config.batch_size)
        .map(normalize_features)
        .map(extract_windows)
        .prefetch(tf.data.AUTOTUNE)
    )
    return pipeline
```

### Architettura del modello

Implementiamo due tipi di modelli per il rilevamento anomalie:

1. **Autoencoder**: Per rilevamento non supervisionato
2. **LSTM-GRU**: Per rilevamento su serie temporali

```python
def build_autoencoder_model(input_shape, encoding_dim=32):
    """Costruisce un modello Autoencoder per anomaly detection."""
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    latent = layers.Dense(encoding_dim, activation='relu')(x)
    
    # Decoder
    x = layers.Dense(64, activation='relu')(latent)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(input_shape[0], activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model
```

### Training e valutazione

Utilizziamo MLflow per tenere traccia degli esperimenti:

```python
def train_model(model, train_data, val_data, config):
    """Addestra il modello con tracking MLflow."""
    with mlflow.start_run():
        mlflow.tensorflow.autolog()
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2
            )
        ]
        
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=config.epochs,
            callbacks=callbacks
        )
        
        return history, model
```

## 🔌 Integrazione Microservizi

### Architettura di comunicazione RabbitMQ

```python
class MessageBroker:
    """Gestisce la comunicazione asincrona tra i servizi."""
    
    def __init__(self, config):
        self.config = config
        self.connection = None
        self.channel = None
        
    async def connect(self):
        """Stabilisce la connessione con RabbitMQ."""
        self.connection = await aio_pika.connect_robust(
            host=self.config.rabbitmq_host,
            port=self.config.rabbitmq_port,
            login=self.config.rabbitmq_user,
            password=self.config.rabbitmq_password
        )
        self.channel = await self.connection.channel()
        
    async def publish(self, exchange_name, routing_key, message):
        """Pubblica un messaggio su un exchange."""
        exchange = await self.channel.declare_exchange(
            exchange_name, 
            aio_pika.ExchangeType.TOPIC,
            durable=True
        )
        
        await exchange.publish(
            aio_pika.Message(
                body=json.dumps(message).encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT
            ),
            routing_key=routing_key
        )
```

### API FastAPI

```python
app = FastAPI(title="PredictSense API", version="1.0.0")

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: PredictionRequest):
    """Endpoint per predizione anomalie in tempo reale."""
    try:
        # Preprocessing
        features = preprocessor.transform(data.features)
        
        # Predizione
        prediction = predictor.predict(features)
        
        # Pubblica risultato
        await message_broker.publish(
            "predictions", 
            "new_prediction", 
            {
                "timestamp": datetime.now().isoformat(),
                "prediction": prediction.tolist(),
                "source": data.source
            }
        )
        
        return {
            "prediction": prediction.tolist(),
            "anomaly_score": float(predictor.get_anomaly_score(prediction)),
            "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## 📊 Dashboard e Visualizzazione

![Dashboard](docs/images/dashboard_preview.png)

La dashboard utilizza Streamlit per visualizzare:
- Dati in tempo reale
- Statistiche di sistema
- Metriche del modello
- Anomalie rilevate

## 🧪 Testing

Esegui i test automatizzati con:

```
pytest tests/
```

Il progetto implementa:
- Test unitari
- Test di integrazione
- Test di carico

## 📚 Risorse di Apprendimento

- [Introduzione a TensorFlow](https://www.tensorflow.org/tutorials)
- [Anomaly Detection con Autoencoders](docs/tutorials/autoencoder_tutorial.md)
- [Architetture a Microservizi](docs/architecture/microservices.md)
- [Best Practices MLOps](docs/mlops/best_practices.md)

## 🤝 Contribuire

Consulta [CONTRIBUTING.md](CONTRIBUTING.md) per linee guida su come contribuire al progetto.

## 📄 Licenza

Distribuito sotto licenza MIT. Vedi [LICENSE](LICENSE) per maggiori informazioni.

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/predictsense&type=Date)](https://star-history.com/#yourusername/predictsense)