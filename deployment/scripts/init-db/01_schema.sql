-- Schema iniziale del database

-- Estensioni
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "timescaledb" CASCADE;

-- Tabella utenti
CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(100) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    role VARCHAR(50) NOT NULL DEFAULT 'user',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Tabella fonti dati
CREATE TABLE IF NOT EXISTS data_sources (
    source_id VARCHAR(100) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    source_type VARCHAR(50) NOT NULL,
    config JSONB NOT NULL DEFAULT '{}'::JSONB,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_by UUID REFERENCES users(user_id)
);

-- Tabella modelli ML
CREATE TABLE IF NOT EXISTS models (
    model_id VARCHAR(100) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    model_type VARCHAR(50) NOT NULL,
    model_path VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'inactive',
    is_default BOOLEAN NOT NULL DEFAULT FALSE,
    config JSONB NOT NULL DEFAULT '{}'::JSONB,
    metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_by UUID REFERENCES users(user_id)
);

-- Tabella predizioni
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id UUID PRIMARY KEY,
    source_id VARCHAR(100) NOT NULL REFERENCES data_sources(source_id),
    model_id VARCHAR(100) NOT NULL REFERENCES models(model_id),
    anomaly_detected BOOLEAN NOT NULL,
    anomaly_score FLOAT NOT NULL,
    confidence FLOAT NOT NULL,
    features JSONB,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Converti predictions in tabella hypertable per time-series ottimizzate
SELECT create_hypertable('predictions', 'timestamp');

-- Tabella dati storici
CREATE TABLE IF NOT EXISTS raw_data (
    data_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id VARCHAR(100) NOT NULL REFERENCES data_sources(source_id),
    data JSONB NOT NULL,
    metadata JSONB,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Converti raw_data in tabella hypertable per time-series ottimizzate
SELECT create_hypertable('raw_data', 'timestamp');

-- Tabella metriche di sistema
CREATE TABLE IF NOT EXISTS system_metrics (
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    service VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Converti system_metrics in tabella hypertable per time-series ottimizzate
SELECT create_hypertable('system_metrics', 'timestamp');

-- Tabella per allarmi
CREATE TABLE IF NOT EXISTS alerts (
    alert_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    prediction_id UUID REFERENCES predictions(prediction_id),
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'new',
    acknowledged_by UUID REFERENCES users(user_id),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Tabella per configurazione alert
CREATE TABLE IF NOT EXISTS alert_configs (
    config_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id VARCHAR(100) NOT NULL REFERENCES data_sources(source_id),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    alert_type VARCHAR(50) NOT NULL,
    conditions JSONB NOT NULL,
    actions JSONB NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_by UUID REFERENCES users(user_id),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Tabella per job di training
CREATE TABLE IF NOT EXISTS training_jobs (
    job_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id VARCHAR(100) REFERENCES models(model_id),
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    config JSONB NOT NULL,
    results JSONB,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_by UUID REFERENCES users(user_id),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indici
CREATE INDEX idx_data_sources_source_type ON data_sources(source_type);
CREATE INDEX idx_models_model_type ON models(model_type);
CREATE INDEX idx_models_status ON models(status);
CREATE INDEX idx_predictions_source_id ON predictions(source_id);
CREATE INDEX idx_predictions_model_id ON predictions(model_id);
CREATE INDEX idx_predictions_anomaly_detected ON predictions(anomaly_detected);
CREATE INDEX idx_alerts_status ON alerts(status);
CREATE INDEX idx_training_jobs_status ON training_jobs(status);

-- Viste
CREATE OR REPLACE VIEW recent_anomalies AS
SELECT 
    p.prediction_id,
    p.source_id,
    ds.name AS source_name,
    p.model_id,
    m.name AS model_name,
    p.anomaly_score,
    p.confidence,
    p.timestamp
FROM 
    predictions p
JOIN 
    data_sources ds ON p.source_id = ds.source_id
JOIN 
    models m ON p.model_id = m.model_id
WHERE 
    p.anomaly_detected = TRUE
ORDER BY 
    p.timestamp DESC;

CREATE OR REPLACE VIEW model_performance AS
SELECT 
    m.model_id,
    m.name,
    m.model_type,
    COUNT(p.prediction_id) AS total_predictions,
    SUM(CASE WHEN p.anomaly_detected THEN 1 ELSE 0 END) AS anomalies_detected,
    AVG(p.confidence) AS avg_confidence,
    m.metrics->>'precision' AS precision,
    m.metrics->>'recall' AS recall,
    m.metrics->>'f1_score' AS f1_score,
    m.metrics->>'auc_roc' AS auc_roc
FROM 
    models m
LEFT JOIN 
    predictions p ON m.model_id = p.model_id
WHERE 
    m.status = 'active'
GROUP BY 
    m.model_id, m.name, m.model_type, m.metrics;

-- Funzioni

-- Funzione per creare un nuovo utente
CREATE OR REPLACE FUNCTION create_user(
    p_username VARCHAR(100),
    p_email VARCHAR(255),
    p_password VARCHAR(255),
    p_first_name VARCHAR(100) DEFAULT NULL,
    p_last_name VARCHAR(100) DEFAULT NULL,
    p_role VARCHAR(50) DEFAULT 'user'
) RETURNS UUID AS $$
DECLARE
    v_user_id UUID;
BEGIN
    INSERT INTO users (
        username, email, password_hash, first_name, last_name, role
    ) VALUES (
        p_username, 
        p_email, 
        crypt(p_password, gen_salt('bf')), 
        p_first_name, 
        p_last_name, 
        p_role
    ) RETURNING user_id INTO v_user_id;
    
    RETURN v_user_id;
END;
$$ LANGUAGE plpgsql;

-- Funzione per autenticare un utente
CREATE OR REPLACE FUNCTION authenticate_user(
    p_username VARCHAR(100),
    p_password VARCHAR(255)
) RETURNS TABLE (
    user_id UUID,
    username VARCHAR(100),
    email VARCHAR(255),
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    role VARCHAR(50)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        u.user_id, u.username, u.email, u.first_name, u.last_name, u.role
    FROM 
        users u
    WHERE 
        u.username = p_username
        AND u.password_hash = crypt(p_password, u.password_hash)
        AND u.is_active = TRUE;
END;
$$ LANGUAGE plpgsql;

-- Funzione per aggiornare un modello come default
CREATE OR REPLACE FUNCTION set_default_model(
    p_model_id VARCHAR(100)
) RETURNS VOID AS $$
BEGIN
    -- Prima rimuovi il default da tutti i modelli
    UPDATE models SET is_default = FALSE;
    
    -- Imposta il nuovo default
    UPDATE models 
    SET is_default = TRUE, updated_at = NOW()
    WHERE model_id = p_model_id;
END;
$$ LANGUAGE plpgsql;

-- Trigger per aggiornare il timestamp 'updated_at'
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_users_timestamp
BEFORE UPDATE ON users
FOR EACH ROW EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER update_data_sources_timestamp
BEFORE UPDATE ON data_sources
FOR EACH ROW EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER update_models_timestamp
BEFORE UPDATE ON models
FOR EACH ROW EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER update_alert_configs_timestamp
BEFORE UPDATE ON alert_configs
FOR EACH ROW EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER update_training_jobs_timestamp
BEFORE UPDATE ON training_jobs
FOR EACH ROW EXECUTE FUNCTION update_timestamp();

