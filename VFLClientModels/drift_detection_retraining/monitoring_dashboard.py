import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from datetime import datetime, timedelta
import numpy as np

class MonitoringDashboard:
    """
    Streamlit dashboard for monitoring VFL model performance and drift
    """
    
    def __init__(self):
        self.setup_page()
    
    def setup_page(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="VFL Credit Scoring - Monitoring Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("VFL Credit Scoring - Monitoring Dashboard")
        st.markdown("---")
    
    def run_dashboard(self):
        """Run the main dashboard"""
        # Sidebar
        self.create_sidebar()
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self.show_overview_metrics()
            self.show_drift_analysis()
            self.show_model_performance()
        
        with col2:
            self.show_alerts()
            self.show_retraining_status()
            self.show_feature_drift()
    
    def create_sidebar(self):
        """Create sidebar with controls"""
        st.sidebar.header("Dashboard Controls")
        
        # Date range selector
        st.sidebar.subheader("📅 Date Range")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        date_range = st.sidebar.date_input(
            "Select date range",
            value=(start_date, end_date),
            max_value=end_date
        )
        
        # Model selector
        st.sidebar.subheader("Model Selection")
        models = ["All Models", "Auto Loans", "Digital Savings", "Home Loans", "Credit Card", "VFL Central"]
        selected_model = st.sidebar.selectbox("Select model", models)
        
        # Drift threshold
        st.sidebar.subheader("⚠️ Drift Thresholds")
        drift_threshold = st.sidebar.slider("Drift threshold", 0.05, 0.5, 0.15, 0.05)
        
        # Refresh button
        if st.sidebar.button("🔄 Refresh Data"):
            st.rerun()
    
    def show_overview_metrics(self):
        """Show overview metrics"""
        st.header("📈 Overview Metrics")
        
        # Create metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Active Models",
                value="5",
                delta="0"
            )
        
        with col2:
            st.metric(
                label="Drift Detected",
                value="2",
                delta="+1",
                delta_color="inverse"
            )
        
        with col3:
            st.metric(
                label="Avg Performance",
                value="94.2%",
                delta="+0.5%"
            )
        
        with col4:
            st.metric(
                label="Last Retraining",
                value="3 days ago",
                delta="-2 days"
            )
    
    def show_drift_analysis(self):
        """Show drift analysis charts"""
        st.header("🔍 Drift Analysis")
        
        # Load drift history
        drift_data = self.load_drift_history()
        
        if drift_data:
            # Create drift timeline
            fig = go.Figure()
            
            dates = [datetime.fromisoformat(d['timestamp']) for d in drift_data]
            drift_detected = [d['overall_drift_detected'] for d in drift_data]
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=drift_detected,
                mode='lines+markers',
                name='Drift Detected',
                line=dict(color='red' if any(drift_detected) else 'green')
            ))
            
            fig.update_layout(
                title="Drift Detection Timeline",
                xaxis_title="Date",
                yaxis_title="Drift Detected",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Drift breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Drift Types")
            drift_types = {
                "Statistical": 3,
                "Performance": 1,
                "Distribution": 2,
                "Prediction": 1
            }
            
            fig = px.pie(
                values=list(drift_types.values()),
                names=list(drift_types.keys()),
                title="Drift Types Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Drift Severity")
            severity_data = pd.DataFrame({
                'Severity': ['Low', 'Medium', 'High'],
                'Count': [5, 2, 1]
            })
            
            fig = px.bar(
                severity_data,
                x='Severity',
                y='Count',
                title="Drift Severity Distribution",
                color='Count',
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def show_model_performance(self):
        """Show model performance metrics"""
        st.header("🎯 Model Performance")
        
        # Performance metrics table
        performance_data = {
            'Model': ['Auto Loans', 'Digital Savings', 'Home Loans', 'Credit Card', 'VFL Central'],
            'MAE': [12.5, 8.2, 15.1, 6.8, 9.3],
            'RMSE': [18.2, 12.1, 22.5, 10.4, 14.7],
            'R²': [0.89, 0.92, 0.85, 0.94, 0.91],
            'Status': ['✅', '⚠️', '✅', '✅', '✅']
        }
        
        df = pd.DataFrame(performance_data)
        st.dataframe(df, use_container_width=True)
        
        # Performance trend
        st.subheader("📈 Performance Trend")
        
        # Simulate performance data over time
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        performance_trend = np.random.normal(0.91, 0.02, len(dates))
        
        fig = px.line(
            x=dates,
            y=performance_trend,
            title="VFL Central Model Performance Trend",
            labels={'x': 'Date', 'y': 'R² Score'}
        )
        
        fig.add_hline(y=0.9, line_dash="dash", line_color="red", 
                     annotation_text="Performance Threshold")
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_alerts(self):
        """Show alerts and notifications"""
        st.header("Alerts")
        
        alerts = [
            {"type": "warning", "message": "Drift detected in Auto Loans model", "time": "2 hours ago"},
            {"type": "error", "message": "Performance degradation in Digital Savings", "time": "1 day ago"},
            {"type": "info", "message": "Scheduled retraining completed", "time": "3 days ago"},
            {"type": "success", "message": "All models performing within thresholds", "time": "1 week ago"}
        ]
        
        for alert in alerts:
            if alert["type"] == "warning":
                st.warning(f"⚠️ {alert['message']} ({alert['time']})")
            elif alert["type"] == "error":
                st.error(f"❌ {alert['message']} ({alert['time']})")
            elif alert["type"] == "info":
                st.info(f"ℹ️ {alert['message']} ({alert['time']})")
            else:
                st.success(f"✅ {alert['message']} ({alert['time']})")
    
    def show_retraining_status(self):
        """Show retraining status"""
        st.header("🔄 Retraining Status")
        
        # Retraining schedule
        st.subheader("Schedule")
        st.write("**Next retraining:** Tomorrow 2:00 AM")
        st.write("**Last retraining:** 3 days ago")
        st.write("**Status:** Scheduled")
        
        # Retraining history
        st.subheader("📋 History")
        retraining_history = [
            {"date": "2024-01-28", "status": "✅ Success", "duration": "45 min"},
            {"date": "2024-01-21", "status": "✅ Success", "duration": "52 min"},
            {"date": "2024-01-14", "status": "⚠️ Partial", "duration": "38 min"}
        ]
        
        for record in retraining_history:
            st.write(f"**{record['date']}:** {record['status']} ({record['duration']})")
    
    def show_feature_drift(self):
        """Show feature-level drift analysis"""
        st.header("🔍 Feature Drift")
        
        # Feature drift table
        feature_drift_data = {
            'Feature': ['income', 'age', 'credit_history', 'employment_length', 'debt_ratio'],
            'Drift Score': [0.12, 0.08, 0.15, 0.06, 0.22],
            'Status': ['⚠️', '✅', '⚠️', '✅', '❌']
        }
        
        df = pd.DataFrame(feature_drift_data)
        st.dataframe(df, use_container_width=True)
        
        # Feature importance vs drift
        st.subheader("Feature Importance vs Drift")
        
        importance = [0.25, 0.20, 0.18, 0.15, 0.12]
        drift_scores = [0.12, 0.08, 0.15, 0.06, 0.22]
        features = ['income', 'age', 'credit_history', 'employment_length', 'debt_ratio']
        
        fig = px.scatter(
            x=importance,
            y=drift_scores,
            text=features,
            title="Feature Importance vs Drift Score",
            labels={'x': 'Feature Importance', 'y': 'Drift Score'}
        )
        
        fig.add_hline(y=0.15, line_dash="dash", line_color="red", 
                     annotation_text="Drift Threshold")
        
        st.plotly_chart(fig, use_container_width=True)
    
    def load_drift_history(self):
        """Load drift detection history"""
        history_path = 'VFLClientModels/data/drift_history.json'
        
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []

def main():
    """Main function to run the dashboard"""
    dashboard = MonitoringDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main() 