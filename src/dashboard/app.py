# src/dashboard/app.py
import os
import json
import logging
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Configurazione logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configurazione ambiente
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "predictsense_api_key")
REFRESH_INTERVAL = int(os.getenv("REFRESH_INTERVAL", "60"))  # in secondi

# Headers per le richieste API
API_HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}


# Funzioni di utilità per chiamate API
def fetch_recent_predictions(limit=100):
    """Recupera le predizioni più recenti."""
    try:
        response = requests.get(
            f"{API_URL}/predictions?limit={limit}",
            headers=API_HEADERS
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching predictions: {str(e)}")
        st.error(f"Failed to fetch predictions: {str(e)}")
        return []


def fetch_anomalies(days=7, limit=500):
    """Recupera le anomalie rilevate negli ultimi giorni."""
    try:
        response = requests.get(
            f"{API_URL}/predictions?anomaly_only=true&limit={limit}",
            headers=API_HEADERS
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching anomalies: {str(e)}")
        st.error(f"Failed to fetch anomalies: {str(e)}")
        return []


def fetch_models():
    """Recupera informazioni sui modelli disponibili."""
    try:
        response = requests.get(
            f"{API_URL}/models",
            headers=API_HEADERS
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching models: {str(e)}")
        st.error(f"Failed to fetch models: {str(e)}")
        return []


def fetch_model_metrics(model_id):
    """Recupera le metriche di un modello specifico."""
    try:
        response = requests.get(
            f"{API_URL}/models/{model_id}/metrics",
            headers=API_HEADERS
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching model metrics: {str(e)}")
        st.error(f"Failed to fetch model metrics: {str(e)}")
        return {}


def fetch_system_metrics():
    """Recupera le metriche di sistema."""
    try:
        response = requests.get(
            f"{API_URL}/metrics/system",
            headers=API_HEADERS
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching system metrics: {str(e)}")
        st.error(f"Failed to fetch system metrics: {str(e)}")
        return {}


# Configurazione della pagina
st.set_page_config(
    page_title="PredictSense Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Sidebar per navigazione
st.sidebar.title("PredictSense")
st.sidebar.caption("Anomaly Detection System")

# Menu principale
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Anomalies", "Models", "System Metrics"]
)

# Informazioni generali
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **PredictSense Dashboard** - version 1.0.0
    
    A machine learning-based anomaly detection system.
    """
)


# Pagina: Overview
if page == "Overview":
    st.title("System Overview")
    
    # Metriche di sistema
    system_metrics = fetch_system_metrics()
    
    if system_metrics:
        st.subheader("System Health")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Active Models",
                value=system_metrics.get("active_models", 0),
                delta=None
            )
        
        with col2:
            st.metric(
                label="Predictions Today",
                value=system_metrics.get("predictions_today", 0),
                delta=f"{system_metrics.get('predictions_change', 0)}%"
            )
        
        with col3:
            st.metric(
                label="Anomalies Today",
                value=system_metrics.get("anomalies_today", 0),
                delta=f"{system_metrics.get('anomalies_change', 0)}%"
            )
        
        with col4:
            st.metric(
                label="Avg Response Time",
                value=f"{system_metrics.get('avg_response_time', 0):.2f} ms",
                delta=None
            )
    
    # Predizioni recenti
    st.subheader("Recent Predictions")
    
    with st.spinner("Loading recent predictions..."):
        predictions = fetch_recent_predictions(limit=100)
    
    if predictions:
        # Converti in DataFrame
        df_predictions = pd.DataFrame(predictions)
        df_predictions['timestamp'] = pd.to_datetime(df_predictions['timestamp'])
        
        # Aggiorna colonna anomaly
        df_predictions['status'] = df_predictions['anomaly_detected'].apply(
            lambda x: "⚠️ Anomaly" if x else "✅ Normal"
        )
        
        # Grafico a linee con punteggi di anomalia
        st.subheader("Anomaly Scores Over Time")
        
        fig = px.line(
            df_predictions.sort_values('timestamp'),
            x='timestamp',
            y='anomaly_score',
            color='status',
            hover_data=['prediction_id', 'confidence', 'processing_time_ms'],
            title="Anomaly Scores Over Time",
            color_discrete_map={"⚠️ Anomaly": "red", "✅ Normal": "green"}
        )
        
        # Aggiungi linea di soglia
        fig.add_shape(
            type="line",
            line=dict(dash="dash", color="orange", width=2),
            y0=0.5, y1=0.5,
            x0=df_predictions['timestamp'].min(),
            x1=df_predictions['timestamp'].max()
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabella
        st.dataframe(
            df_predictions[['prediction_id', 'timestamp', 'status', 'anomaly_score', 'confidence']],
            use_container_width=True
        )
    else:
        st.info("No predictions found.")


# Pagina: Anomalies
elif page == "Anomalies":
    st.title("Detected Anomalies")
    
    time_range = st.selectbox(
        "Time Range",
        ["Last 24 Hours", "Last 7 Days", "Last 30 Days"],
        index=1
    )
    
    days_map = {
        "Last 24 Hours": 1,
        "Last 7 Days": 7,
        "Last 30 Days": 30
    }
    
    days = days_map[time_range]
    
    with st.spinner(f"Loading anomalies for the {time_range.lower()}..."):
        anomalies = fetch_anomalies(days=days)
    
    if anomalies:
        # Converti in DataFrame
        df_anomalies = pd.DataFrame(anomalies)
        df_anomalies['timestamp'] = pd.to_datetime(df_anomalies['timestamp'])
        
        # Filtro per data
        cutoff_date = datetime.now() - timedelta(days=days)
        df_anomalies = df_anomalies[df_anomalies['timestamp'] >= cutoff_date]
        
        # Visualizzazioni
        st.subheader(f"Anomalies - {time_range}")
        
        # Heatmap di anomalie per ora del giorno
        df_anomalies['hour'] = df_anomalies['timestamp'].dt.hour
        df_anomalies['day'] = df_anomalies['timestamp'].dt.date
        
        # Grafico a dispersione per anomalie
        st.subheader("Anomalies by Severity and Time")
        
        fig = px.scatter(
            df_anomalies,
            x='timestamp',
            y='anomaly_score',
            size='confidence',
            color='anomaly_score',
            hover_data=['prediction_id', 'model_id'],
            title="Anomalies by Severity and Time",
            color_continuous_scale="Reds"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribuzione punteggi
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist = px.histogram(
                df_anomalies,
                x='anomaly_score',
                nbins=20,
                title="Distribution of Anomaly Scores",
                color_discrete_sequence=['red']
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            if 'model_id' in df_anomalies.columns:
                model_counts = df_anomalies['model_id'].value_counts()
                
                fig_pie = px.pie(
                    names=model_counts.index,
                    values=model_counts.values,
                    title="Anomalies by Model"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        # Tabella dettagliata
        st.subheader("Anomaly Details")
        st.dataframe(
            df_anomalies[['prediction_id', 'timestamp', 'anomaly_score', 'confidence', 'model_id']],
            use_container_width=True
        )
    else:
        st.info(f"No anomalies found in the {time_range.lower()}.")


# Pagina: Models
elif page == "Models":
    st.title("Model Performance")
    
    with st.spinner("Loading models..."):
        models = fetch_models()
    
    if models:
        # Selezione modello
        model_options = {m['model_id']: f"{m['model_id']} - {m['model_type']}" for m in models}
        
        selected_model_id = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x]
        )
        
        # Trova il modello selezionato
        selected_model = next(m for m in models if m['model_id'] == selected_model_id)
        
        # Dettagli modello
        st.subheader("Model Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Model Type",
                value=selected_model['model_type']
            )
        
        with col2:
            st.metric(
                label="Status",
                value=selected_model['status']
            )
        
        with col3:
            st.metric(
                label="Last Updated",
                value=selected_model.get('last_updated', 'N/A').split('T')[0]
            )
        
        # Metrics
        with st.spinner("Loading model metrics..."):
            metrics = fetch_model_metrics(selected_model_id)
        
        if metrics:
            st.subheader("Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Precision",
                    value=f"{metrics.get('precision', 0):.2f}",
                    delta=None
                )
            
            with col2:
                st.metric(
                    label="Recall",
                    value=f"{metrics.get('recall', 0):.2f}",
                    delta=None
                )
            
            with col3:
                st.metric(
                    label="F1 Score",
                    value=f"{metrics.get('f1_score', 0):.2f}",
                    delta=None
                )
            
            with col4:
                st.metric(
                    label="AUC-ROC",
                    value=f"{metrics.get('auc_roc', 0):.2f}",
                    delta=None
                )
            
            # Confusion Matrix
            if 'confusion_matrix' in metrics:
                st.subheader("Confusion Matrix")
                
                cm = metrics['confusion_matrix']
                
                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Predicted Normal', 'Predicted Anomaly'],
                    y=['Actual Normal', 'Actual Anomaly'],
                    colorscale='Blues',
                    showscale=False
                ))
                
                fig.update_layout(
                    title='Confusion Matrix',
                    xaxis=dict(title='Predicted Class'),
                    yaxis=dict(title='Actual Class')
                )
                
                # Aggiungi annotazioni
                annotations = []
                for i, row in enumerate(cm):
                    for j, value in enumerate(row):
                        annotations.append(
                            dict(
                                x=j,
                                y=i,
                                text=str(value),
                                font=dict(color='white' if value > (max(map(max, cm)) / 2) else 'black'),
                                showarrow=False
                            )
                        )
                
                fig.update_layout(annotations=annotations)
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Training History
            if 'training_history' in metrics:
                st.subheader("Training History")
                
                history = metrics['training_history']
                
                if isinstance(history, list) and len(history) > 0:
                    # Converti a DataFrame
                    df_history = pd.DataFrame(history)
                    
                    # Plot delle metriche
                    fig = go.Figure()
                    
                    for column in df_history.columns:
                        if column != 'epoch':
                            fig.add_trace(go.Scatter(
                                x=df_history['epoch'],
                                y=df_history[column],
                                mode='lines+markers',
                                name=column
                            ))
                    
                    fig.update_layout(
                        title='Training Metrics Over Epochs',
                        xaxis_title='Epoch',
                        yaxis_title='Value',
                        legend_title='Metric'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Feature Importance
            if 'feature_importance' in metrics:
                st.subheader("Feature Importance")
                
                features = metrics['feature_importance']
                
                if features and isinstance(features, dict):
                    df_features = pd.DataFrame({
                        'Feature': list(features.keys()),
                        'Importance': list(features.values())
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(
                        df_features,
                        x='Feature',
                        y='Importance',
                        title="Feature Importance",
                        color='Importance',
                        color_continuous_scale='Blues'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("No metrics available for this model.")
    
    else:
        st.info("No models found.")


# Pagina: System Metrics
elif page == "System Metrics":
    st.title("System Metrics")
    
    with st.spinner("Loading system metrics..."):
        system_metrics = fetch_system_metrics()
    
    if system_metrics:
        # Metriche principali
        st.subheader("Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="CPU Usage",
                value=f"{system_metrics.get('cpu_usage', 0):.1f}%",
                delta=None
            )
        
        with col2:
            st.metric(
                label="Memory Usage",
                value=f"{system_metrics.get('memory_usage', 0):.1f}%",
                delta=None
            )
        
        with col3:
            st.metric(
                label="Disk Usage",
                value=f"{system_metrics.get('disk_usage', 0):.1f}%",
                delta=None
            )
        
        with col4:
            st.metric(
                label="Uptime",
                value=f"{system_metrics.get('uptime_days', 0)} days",
                delta=None
            )
        
        # Timeline di utilizzo
        if 'usage_history' in system_metrics:
            st.subheader("Resource Usage History")
            
            usage_history = system_metrics['usage_history']
            
            if isinstance(usage_history, list) and len(usage_history) > 0:
                # Converti a DataFrame
                df_usage = pd.DataFrame(usage_history)
                df_usage['timestamp'] = pd.to_datetime(df_usage['timestamp'])
                
                # Plot CPU, Memory, Disk
                fig = go.Figure()
                
                if 'cpu_usage' in df_usage.columns:
                    fig.add_trace(go.Scatter(
                        x=df_usage['timestamp'],
                        y=df_usage['cpu_usage'],
                        mode='lines',
                        name='CPU Usage %'
                    ))
                
                if 'memory_usage' in df_usage.columns:
                    fig.add_trace(go.Scatter(
                        x=df_usage['timestamp'],
                        y=df_usage['memory_usage'],
                        mode='lines',
                        name='Memory Usage %'
                    ))
                
                if 'disk_usage' in df_usage.columns:
                    fig.add_trace(go.Scatter(
                        x=df_usage['timestamp'],
                        y=df_usage['disk_usage'],
                        mode='lines',
                        name='Disk Usage %'
                    ))
                
                fig.update_layout(
                    title='Resource Usage History',
                    xaxis_title='Time',
                    yaxis_title='Usage %',
                    legend_title='Resource'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Service Health
        if 'service_health' in system_metrics:
            st.subheader("Service Health")
            
            services = system_metrics['service_health']
            
            if isinstance(services, dict):
                # Converti a DataFrame
                df_services = pd.DataFrame([
                    {"Service": service, "Status": status}
                    for service, status in services.items()
                ])
                
                # Visualizzazione status servizi
                for _, row in df_services.iterrows():
                    status_color = "green" if row['Status'] == "healthy" else "red"
                    st.markdown(
                        f"<div style='display: flex; align-items: center; margin-bottom: 10px;'>"
                        f"<div style='width: 10px; height: 10px; border-radius: 50%; background-color: {status_color}; margin-right: 10px;'></div>"
                        f"<div><strong>{row['Service']}</strong>: {row['Status']}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
        
        # Metriche di rete
        if 'network_metrics' in system_metrics:
            st.subheader("Network Metrics")
            
            network = system_metrics['network_metrics']
            
            if isinstance(network, dict):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Requests/Second",
                        value=f"{network.get('requests_per_second', 0):.2f}",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        label="Avg Response Time",
                        value=f"{network.get('avg_response_time', 0):.2f} ms",
                        delta=None
                    )
                
                with col3:
                    st.metric(
                        label="Error Rate",
                        value=f"{network.get('error_rate', 0):.2f}%",
                        delta=None
                    )
    
    else:
        st.info("No system metrics available.")


# Footer
st.sidebar.markdown("---")
st.sidebar.caption("© 2025 PredictSense")