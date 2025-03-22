-- Dati iniziali per il database

-- Utente admin
SELECT create_user(
    'admin',
    'admin@predictsense.com',
    'admin123',
    'Admin',
    'User',
    'admin'
);

-- Fonte dati di esempio
INSERT INTO data_sources (
    source_id, name, description, source_type, config, enabled
) VALUES (
    'network_traffic',
    'Network Traffic Analyzer',
    'Monitors network traffic patterns for anomalies',
    'network',
    '{"collection_interval": 60, "metrics": ["bytes_sent", "bytes_received", "packet_loss", "latency", "connection_count"]}',
    TRUE
);

INSERT INTO data_sources (
    source_id, name, description, source_type, config, enabled
) VALUES (
    'system_metrics',
    'Server Metrics',
    'Monitors system metrics like CPU, memory and disk usage',
    'system',
    '{"collection_interval": 30, "metrics": ["cpu_usage", "memory_usage", "disk_usage", "io_wait", "load_average"]}',
    TRUE
);

INSERT INTO data_sources (
    source_id, name, description, source_type, config, enabled
) VALUES (
    'application_logs',
    'Application Logs Analyzer',
    'Analyzes application logs for anomalous patterns',
    'logs',
    '{"collection_interval": 300, "log_path": "/var/log/application", "pattern": "ERROR|WARN"}',
    TRUE
);

-- Modelli di esempio
INSERT INTO models (
    model_id, name, description, model_type, model_path, version, status, is_default, metrics
) VALUES (
    'autoencoder_v1',
    'Network Traffic Autoencoder',
    'Autoencoder model for network traffic anomaly detection',
    'autoencoder',
    'network/autoencoder_v1',
    '1.0',
    'active',
    TRUE,
    '{"precision": 0.92, "recall": 0.89, "f1_score": 0.91, "auc_roc": 0.95}'
);

INSERT INTO models (
    model_id, name, description, model_type, model_path, version, status, metrics
) VALUES (
    'lstm_system_v1',
    'System Metrics LSTM',
    'LSTM model for system metrics time-series anomaly detection',
    'lstm',
    'system/lstm_v1',
    '1.0',
    'active',
    '{"precision": 0.88, "recall": 0.92, "f1_score": 0.90, "auc_roc": 0.94}'
);

INSERT INTO models (
    model_id, name, description, model_type, model_path, version, status, metrics
) VALUES (
    'ensemble_v1',
    'Multi-source Ensemble',
    'Ensemble model combining multiple detectors for more robust anomaly detection',
    'ensemble',
    'ensemble/v1',
    '1.0',
    'inactive',
    '{"precision": 0.94, "recall": 0.91, "f1_score": 0.93, "auc_roc": 0.97}'
);

-- Configurazioni di allarme
INSERT INTO alert_configs (
    source_id, name, description, alert_type, conditions, actions, enabled
) VALUES (
    'network_traffic',
    'High Anomaly Score Alert',
    'Triggers when a high anomaly score is detected in network traffic',
    'anomaly_score',
    '{"threshold": 0.85, "operator": "gt", "window": 300}',
    '{"email": ["alerts@predictsense.com"], "webhook": ["https://hooks.slack.com/services/XXX/YYY/ZZZ"]}',
    TRUE
);

INSERT INTO alert_configs (
    source_id, name, description, alert_type, conditions, actions, enabled
) VALUES (
    'system_metrics',
    'Consecutive Anomalies Alert',
    'Triggers when multiple consecutive anomalies are detected in system metrics',
    'consecutive_anomalies',
    '{"count": 3, "window": 600}',
    '{"email": ["alerts@predictsense.com"], "sms": ["+1234567890"]}',
    TRUE
);