import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional


def render_overview(predictions: List[Dict], system_metrics: Dict):
    """Renderizza la pagina di overview."""
    st.title("System Overview")
    
    # Metriche di sistema
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