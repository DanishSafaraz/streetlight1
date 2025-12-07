# dashboard_realtime.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os
import time
import json
from datetime import datetime, timedelta
import threading
import queue
import paho.mqtt.client as mqtt
import ssl
import asyncio
import warnings
warnings.filterwarnings('ignore')

# =============================================
# MQTT CONFIGURATION
# =============================================
MQTT_BROKER = "48be83e63863499c87afce855025c93e.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_DATA_TOPIC = "iot/streetlight/data"
MQTT_PREDICT_TOPIC = "iot/predict/streetlight"
MQTT_USERNAME = "hivemq.webclient.1765084748576"
MQTT_PASSWORD = "y9!uH7$Id8La4>PDCm.h"

# =============================================
# GLOBAL VARIABLES
# =============================================
latest_data = None
data_history = []
predictions_history = []
mqtt_connected = False
data_queue = queue.Queue()
stop_threads = False

# =============================================
# MQTT FUNCTIONS
# =============================================
class MQTTClient:
    def __init__(self):
        self.client = None
        self.connected = False
        self.data_buffer = []
        self.max_buffer_size = 100
        
    def connect(self):
        """Connect to MQTT broker"""
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, protocol=mqtt.MQTTv5)
            self.client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
            
            # Setup TLS
            self.client.tls_set(cert_reqs=ssl.CERT_NONE)
            self.client.tls_insecure_set(True)
            
            # Set callbacks
            self.client.on_connect = self.on_connect
            self.client.on_message = self.on_message
            self.client.on_disconnect = self.on_disconnect
            
            # Connect
            self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.client.loop_start()
            
            return True
        except Exception as e:
            st.error(f"MQTT Connection Error: {e}")
            return False
    
    def on_connect(self, client, userdata, flags, rc, properties=None):
        """Callback when connected to MQTT"""
        global mqtt_connected
        if rc == 0:
            self.connected = True
            mqtt_connected = True
            st.session_state.mqtt_status = "🟢 Connected"
            client.subscribe(MQTT_DATA_TOPIC)
            print(f"✅ Subscribed to {MQTT_DATA_TOPIC}")
        else:
            self.connected = False
            mqtt_connected = False
            st.session_state.mqtt_status = f"🔴 Connection failed: {rc}"
    
    def on_message(self, client, userdata, msg):
        """Callback when message received"""
        try:
            payload = msg.payload.decode('utf-8')
            data = self.parse_mqtt_payload(payload)
            
            if data:
                # Add to global variables
                global latest_data, data_history
                latest_data = data
                data_history.append(data)
                
                # Keep only last 100 entries
                if len(data_history) > 100:
                    data_history = data_history[-100:]
                
                # Put in queue for processing
                data_queue.put(data)
                
                # Update session state for Streamlit
                st.session_state.latest_data = data
                st.session_state.data_history = data_history[-20:]  # Last 20 for display
                
        except Exception as e:
            print(f"Error processing MQTT message: {e}")
    
    def parse_mqtt_payload(self, payload):
        """Parse MQTT payload: {timestamp;intensity;voltage}"""
        try:
            # Remove curly braces
            payload = payload.strip('{}')
            parts = payload.split(';')
            
            if len(parts) >= 3:
                timestamp = parts[0].strip()
                intensity = float(parts[1].strip())
                voltage = float(parts[2].strip())
                
                # Parse timestamp
                try:
                    dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                except:
                    dt = datetime.now()
                
                # Create data dict
                data = {
                    'timestamp': dt,
                    'timestamp_str': timestamp,
                    'light_intensity': intensity,
                    'voltage': voltage,
                    'lamp_status': 'ON' if voltage > 0 else 'OFF'
                }
                
                return data
        except Exception as e:
            print(f"Error parsing payload: {e}")
        return None
    
    def publish_prediction(self, prediction_data):
        """Publish prediction to MQTT"""
        if self.connected and self.client:
            try:
                # Format: {timestamp;model;prediction;confidence;features}
                payload = json.dumps(prediction_data)
                result = self.client.publish(MQTT_PREDICT_TOPIC, payload, qos=1)
                
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    return True
                else:
                    return False
            except Exception as e:
                st.error(f"Publish error: {e}")
                return False
        return False
    
    def on_disconnect(self, client, userdata, rc, properties=None):
        """Callback when disconnected"""
        self.connected = False
        global mqtt_connected
        mqtt_connected = False
        st.session_state.mqtt_status = "🔴 Disconnected"
    
    def disconnect(self):
        """Disconnect from MQTT"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()

# =============================================
# STREAMLIT PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="Streetlight Realtime Dashboard",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .realtime-card {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .status-connected {
        color: #4CAF50;
        font-weight: bold;
    }
    .status-disconnected {
        color: #F44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'latest_data' not in st.session_state:
    st.session_state.latest_data = None
if 'data_history' not in st.session_state:
    st.session_state.data_history = []
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'mqtt_status' not in st.session_state:
    st.session_state.mqtt_status = "🔴 Disconnected"
if 'mqtt_client' not in st.session_state:
    st.session_state.mqtt_client = None

# =============================================
# ML MODEL FUNCTIONS
# =============================================
@st.cache_resource
def load_ml_models():
    """Load ML models and preprocessing objects"""
    try:
        models_path = r"\Trainingmodel\models_hard"
        
        # Load models
        models = {
            'Decision Tree': joblib.load(os.path.join(models_path, "decision_tree.pkl")),
            'K-Nearest Neighbors': joblib.load(os.path.join(models_path, "k-nearest_neighbors.pkl")),
            'Logistic Regression': joblib.load(os.path.join(models_path, "logistic_regression.pkl"))
        }
        
        # Load preprocessing
        scaler = joblib.load(os.path.join(models_path, "feature_scaler.pkl"))
        encoder = joblib.load(os.path.join(models_path, "target_encoder.pkl"))
        
        # Load comparison data
        comparison_df = pd.read_csv(os.path.join(models_path, "model_comparison_detailed.csv"))
        
        return {
            'models': models,
            'scaler': scaler,
            'encoder': encoder,
            'comparison': comparison_df
        }
    except Exception as e:
        st.error(f"Error loading ML models: {e}")
        return None

def prepare_features(data):
    """Prepare features from incoming data for ML prediction"""
    try:
        # Extract base features
        intensity = data['light_intensity']
        voltage = data['voltage']
        timestamp = data['timestamp']
        
        # Create time-based features
        hour = timestamp.hour
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        # Create additional features (simulated - would need real calculations)
        # In real system, these would come from historical data
        intensity_rolling_mean = intensity  # Placeholder
        intensity_rolling_std = 0.0  # Placeholder
        lamp_on_duration = 1 if voltage > 0 else 0  # Placeholder
        energy_estimate = voltage * (100 - intensity) / 10000
        
        # Features must match training features
        features = np.array([[
            intensity, voltage, hour, hour_sin, hour_cos,
            intensity_rolling_mean, intensity_rolling_std,
            lamp_on_duration, energy_estimate
        ]])
        
        return features
    except Exception as e:
        st.error(f"Error preparing features: {e}")
        return None

def make_prediction(data, ml_data):
    """Make prediction using all models"""
    try:
        features = prepare_features(data)
        if features is None:
            return None
        
        # Scale features
        features_scaled = ml_data['scaler'].transform(features)
        
        predictions = []
        
        for model_name, model in ml_data['models'].items():
            try:
                # Predict
                pred_encoded = model.predict(features_scaled)[0]
                pred_class = ml_data['encoder'].inverse_transform([pred_encoded])[0]
                
                # Get confidence
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(features_scaled)[0]
                    confidence = float(np.max(proba)) * 100
                    proba_dict = {ml_data['encoder'].classes_[i]: float(proba[i]) 
                                 for i in range(len(proba))}
                else:
                    confidence = 100.0
                    proba_dict = {}
                
                prediction = {
                    'timestamp': data['timestamp'],
                    'timestamp_str': data['timestamp_str'],
                    'model': model_name,
                    'prediction': pred_class,
                    'confidence': confidence,
                    'probabilities': proba_dict,
                    'input_intensity': data['light_intensity'],
                    'input_voltage': data['voltage'],
                    'input_hour': data['timestamp'].hour
                }
                
                predictions.append(prediction)
                
            except Exception as e:
                st.warning(f"Prediction error for {model_name}: {e}")
        
        return predictions
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# =============================================
# STREAMLIT UI
# =============================================
# Title
st.markdown('<h1 class="main-header">💡 Streetlight Realtime Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # MQTT Status
    status_emoji = "🟢" if st.session_state.mqtt_status.startswith("🟢") else "🔴"
    st.metric("MQTT Status", st.session_state.mqtt_status.split(" ")[1] if " " in st.session_state.mqtt_status else st.session_state.mqtt_status)
    
    # Connection controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Connect MQTT", type="primary"):
            if st.session_state.mqtt_client is None:
                mqtt_client = MQTTClient()
                if mqtt_client.connect():
                    st.session_state.mqtt_client = mqtt_client
                    st.success("MQTT Connected!")
                    st.rerun()
    
    with col2:
        if st.button("Disconnect"):
            if st.session_state.mqtt_client:
                st.session_state.mqtt_client.disconnect()
                st.session_state.mqtt_client = None
                st.session_state.mqtt_status = "🔴 Disconnected"
                st.rerun()
    
    st.markdown("---")
    
    # Model selection
    st.subheader("🎯 Model Settings")
    selected_model = st.selectbox(
        "Primary Model for Display",
        ["All Models", "Decision Tree", "K-Nearest Neighbors", "Logistic Regression"]
    )
    
    # Prediction settings
    auto_predict = st.checkbox("Auto-predict on new data", value=True)
    publish_predictions = st.checkbox("Publish predictions to MQTT", value=True)
    
    st.markdown("---")
    
    # System info
    st.caption("**System Information**")
    st.write(f"📡 Topic: {MQTT_DATA_TOPIC}")
    st.write(f"📤 Predict Topic: {MQTT_PREDICT_TOPIC}")
    st.write(f"🤖 Models: 3 loaded")
    
    st.markdown("---")
    st.caption("Realtime Streetlight Monitoring v1.0")

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Realtime Monitor", 
    "🔮 Predictions", 
    "📈 Analytics", 
    "⚙️ System"
])

# Load ML models
ml_data = load_ml_models()

with tab1:
    st.header("📊 Realtime Streetlight Monitor")
    
    # Connection status
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        status_class = "status-connected" if st.session_state.mqtt_status.startswith("🟢") else "status-disconnected"
        st.markdown(f"<p class='{status_class}'>{st.session_state.mqtt_status}</p>", unsafe_allow_html=True)
    
    with col2:
        data_count = len(st.session_state.data_history)
        st.metric("Data Points", data_count)
    
    with col3:
        if st.session_state.latest_data:
            last_update = st.session_state.latest_data['timestamp_str']
            st.metric("Last Update", last_update.split(" ")[1])
        else:
            st.metric("Last Update", "No data")
    
    with col4:
        if st.session_state.mqtt_client and st.session_state.mqtt_client.connected:
            st.success("Active")
        else:
            st.error("Inactive")
    
    # Realtime data display
    st.subheader("🔄 Live Data Stream")
    
    # Create placeholder for live updates
    data_placeholder = st.empty()
    chart_placeholder = st.empty()
    
    # Display latest data
    if st.session_state.latest_data:
        data = st.session_state.latest_data
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="realtime-card">
                <h3>💡 Light Intensity</h3>
                <h2>{data['light_intensity']:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            status_color = "#4CAF50" if data['voltage'] > 0 else "#F44336"
            status_text = "ON" if data['voltage'] > 0 else "OFF"
            st.markdown(f"""
            <div class="realtime-card" style="background: linear-gradient(135deg, {status_color} 0%, #FF9800 100%);">
                <h3>⚡ Lamp Status</h3>
                <h2>{status_text}</h2>
                <p>{data['voltage']}V</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="realtime-card">
                <h3>🕒 Time</h3>
                <h4>{data['timestamp_str']}</h4>
            </div>
            """, unsafe_allow_html=True)
        
        # Time series chart
        if len(st.session_state.data_history) > 1:
            history_df = pd.DataFrame(st.session_state.data_history[-50:])  # Last 50 points
            
            fig = go.Figure()
            
            # Light intensity line
            fig.add_trace(go.Scatter(
                x=history_df['timestamp'],
                y=history_df['light_intensity'],
                mode='lines+markers',
                name='Light Intensity',
                line=dict(color='#FF9800', width=2)
            ))
            
            # Voltage as bars
            fig.add_trace(go.Bar(
                x=history_df['timestamp'],
                y=history_df['voltage'],
                name='Voltage',
                marker_color='#1E88E5',
                opacity=0.6,
                yaxis='y2'
            ))
            
            fig.update_layout(
                title='Realtime Sensor Data',
                xaxis_title='Time',
                yaxis_title='Light Intensity (%)',
                yaxis2=dict(
                    title='Voltage (V)',
                    overlaying='y',
                    side='right'
                ),
                hovermode='x unified',
                height=400
            )
            
            chart_placeholder.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Waiting for data from MQTT...")
        # Show sample data for demo
        sample_data = {
            'timestamp': datetime.now(),
            'timestamp_str': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'light_intensity': 65.5,
            'voltage': 220.0,
            'lamp_status': 'ON'
        }
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("💡 Light Intensity (Sample)", "65.5%")
        with col2:
            st.metric("⚡ Voltage (Sample)", "220V")
        
        st.caption("Sample data - connect to MQTT for real data")

with tab2:
    st.header("🔮 ML Predictions")
    
    if ml_data is None:
        st.error("ML models not loaded. Check models_hard folder.")
    else:
        # Display model performance
        st.subheader("Model Performance")
        
        comparison_df = ml_data['comparison']
        best_model = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best Model", best_model['Model'])
        with col2:
            st.metric("Best Accuracy", f"{best_model['Accuracy']:.2%}")
        with col3:
            st.metric("Avg F1-Score", f"{comparison_df['F1-Score'].mean():.2%}")
        
        # Prediction area
        st.subheader("Live Predictions")
        
        if st.session_state.latest_data and auto_predict:
            data = st.session_state.latest_data
            predictions = make_prediction(data, ml_data)
            
            if predictions:
                # Store predictions
                for pred in predictions:
                    if pred not in st.session_state.predictions:
                        st.session_state.predictions.append(pred)
                
                # Keep only last 20 predictions
                if len(st.session_state.predictions) > 20:
                    st.session_state.predictions = st.session_state.predictions[-20:]
                
                # Display predictions
                pred_df = pd.DataFrame(predictions)
                
                # Show current predictions
                st.write("**Current Predictions:**")
                
                for pred in predictions:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Model", pred['model'])
                    with col2:
                        st.metric("Prediction", pred['prediction'])
                    with col3:
                        st.metric("Confidence", f"{pred['confidence']:.1f}%")
                    with col4:
                        if st.button("📤 Publish", key=f"pub_{pred['model']}"):
                            if st.session_state.mqtt_client:
                                # Prepare prediction data for MQTT
                                mqtt_data = {
                                    'timestamp': pred['timestamp_str'],
                                    'model': pred['model'],
                                    'prediction': pred['prediction'],
                                    'confidence': pred['confidence'],
                                    'intensity': pred['input_intensity'],
                                    'voltage': pred['input_voltage'],
                                    'hour': pred['input_hour']
                                }
                                
                                if st.session_state.mqtt_client.publish_prediction(mqtt_data):
                                    st.success(f"Published {pred['model']} prediction")
                                else:
                                    st.error("Publish failed")
                            else:
                                st.error("MQTT not connected")
                
                # Publish all predictions if enabled
                if publish_predictions and st.session_state.mqtt_client:
                    for pred in predictions:
                        mqtt_data = {
                            'timestamp': pred['timestamp_str'],
                            'model': pred['model'],
                            'prediction': pred['prediction'],
                            'confidence': pred['confidence']
                        }
                        st.session_state.mqtt_client.publish_prediction(mqtt_data)
                
                # Prediction history chart
                if len(st.session_state.predictions) > 1:
                    st.subheader("Prediction History")
                    
                    history_df = pd.DataFrame(st.session_state.predictions)
                    
                    # Group by model and prediction
                    pivot_df = history_df.pivot_table(
                        index='timestamp',
                        columns='model',
                        values='prediction',
                        aggfunc='first'
                    ).fillna(method='ffill')
                    
                    fig = px.line(pivot_df, title='Prediction History by Model')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No predictions generated")
        else:
            st.info("Waiting for data to generate predictions...")
            
            # Manual prediction option
            st.subheader("Manual Prediction")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                manual_intensity = st.slider("Light Intensity", 0, 100, 65)
            with col2:
                manual_voltage = st.selectbox("Voltage", [0, 220], format_func=lambda x: f"{x}V")
            with col3:
                manual_hour = st.slider("Hour", 0, 23, 14)
            
            if st.button("Predict Manually"):
                manual_data = {
                    'timestamp': datetime.now(),
                    'timestamp_str': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'light_intensity': manual_intensity,
                    'voltage': manual_voltage
                }
                
                predictions = make_prediction(manual_data, ml_data)
                if predictions:
                    for pred in predictions:
                        st.success(f"{pred['model']}: {pred['prediction']} ({pred['confidence']:.1f}% confidence)")

with tab3:
    st.header("📈 Data Analytics")
    
    if len(st.session_state.data_history) > 0:
        # Convert to DataFrame
        df = pd.DataFrame(st.session_state.data_history)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Intensity", f"{df['light_intensity'].mean():.1f}%")
        with col2:
            st.metric("Max Intensity", f"{df['light_intensity'].max():.1f}%")
        with col3:
            on_percentage = (df['voltage'] > 0).mean() * 100
            st.metric("Lamp ON %", f"{on_percentage:.1f}%")
        with col4:
            st.metric("Data Points", len(df))
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Intensity distribution
            fig1 = px.histogram(df, x='light_intensity', 
                               title='Light Intensity Distribution',
                               nbins=20)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Voltage status
            status_counts = df['lamp_status'].value_counts()
            fig2 = px.pie(values=status_counts.values, 
                         names=status_counts.index,
                         title='Lamp Status Distribution')
            st.plotly_chart(fig2, use_container_width=True)
        
        # Time-based analysis
        st.subheader("Time-based Analysis")
        
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            
            # Group by hour
            hourly_stats = df.groupby('hour').agg({
                'light_intensity': 'mean',
                'voltage': lambda x: (x > 0).mean() * 100
            }).reset_index()
            
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=hourly_stats['hour'], 
                                     y=hourly_stats['light_intensity'],
                                     name='Avg Intensity',
                                     line=dict(color='orange')))
            fig3.add_trace(go.Scatter(x=hourly_stats['hour'], 
                                     y=hourly_stats['voltage'],
                                     name='Lamp ON %',
                                     line=dict(color='blue'),
                                     yaxis='y2'))
            
            fig3.update_layout(
                title='Hourly Patterns',
                xaxis_title='Hour of Day',
                yaxis_title='Light Intensity (%)',
                yaxis2=dict(title='Lamp ON %', overlaying='y', side='right'),
                height=400
            )
            
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Collect more data to see analytics")

with tab4:
    st.header("⚙️ System Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("MQTT Settings")
        st.write(f"**Broker:** {MQTT_BROKER}")
        st.write(f"**Port:** {MQTT_PORT}")
        st.write(f"**Data Topic:** {MQTT_DATA_TOPIC}")
        st.write(f"**Predict Topic:** {MQTT_PREDICT_TOPIC}")
        
        # Test connection
        if st.button("Test MQTT Connection"):
            if st.session_state.mqtt_client and st.session_state.mqtt_client.connected:
                st.success("✅ MQTT Connection Active")
            else:
                st.error("❌ MQTT Not Connected")
    
    with col2:
        st.subheader("ML Models")
        if ml_data:
            st.success("✅ Models Loaded Successfully")
            st.write(f"**Models:** {len(ml_data['models'])}")
            st.write(f"**Classes:** {list(ml_data['encoder'].classes_)}")
            
            # Model info
            for model_name in ml_data['models'].keys():
                model_acc = comparison_df[comparison_df['Model'] == model_name]['Accuracy'].values
                if len(model_acc) > 0:
                    st.write(f"- {model_name}: {model_acc[0]:.2%}")
        else:
            st.error("❌ Models Not Loaded")
    
    # Data management
    st.subheader("Data Management")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Data History"):
            st.session_state.data_history = []
            st.session_state.predictions = []
            st.session_state.latest_data = None
            st.success("Data cleared!")
            st.rerun()
    
    with col2:
        if st.button("Export Data"):
            if len(st.session_state.data_history) > 0:
                df = pd.DataFrame(st.session_state.data_history)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"streetlight_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No data to export")
    
    # System logs
    st.subheader("System Logs")
    log_placeholder = st.empty()
    
    # Simulate logs
    logs = [
        f"{datetime.now().strftime('%H:%M:%S')} - Dashboard started",
        f"{datetime.now().strftime('%H:%M:%S')} - ML models loaded: 3 models",
        f"{datetime.now().strftime('%H:%M:%S')} - MQTT: {st.session_state.mqtt_status}",
    ]
    
    if st.session_state.latest_data:
        logs.append(f"{datetime.now().strftime('%H:%M:%S')} - Latest data: {st.session_state.latest_data['light_intensity']}%")
    
    log_text = "\n".join(logs[-10:])  # Last 10 logs
    log_placeholder.text_area("Recent Logs", log_text, height=150)

# =============================================
# AUTO-REFRESH AND CLEANUP
# =============================================
# Auto-refresh for realtime updates
if st.session_state.mqtt_status.startswith("🟢"):
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p><strong>💡 Smart Streetlight Realtime Monitoring System</strong></p>
    <p>ML Predictions • MQTT Realtime • Streamlit Dashboard</p>
    <p style="font-size: 0.8rem;">
        Data: {data_count} points | Predictions: {pred_count} | Last update: {last_update}
    </p>
</div>
""".format(
    data_count=len(st.session_state.data_history),
    pred_count=len(st.session_state.predictions),
    last_update=datetime.now().strftime("%H:%M:%S")
), unsafe_allow_html=True)

# =============================================
# CLEANUP ON EXIT
# =============================================
def cleanup():
    """Cleanup resources on exit"""
    if st.session_state.mqtt_client:
        st.session_state.mqtt_client.disconnect()

# Register cleanup
import atexit

atexit.register(cleanup)
