import streamlit as st
import requests
import json
from datetime import datetime
import pandas as pd

# Configure the page
st.set_page_config(
    page_title="Credit Score Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .score-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .explanation-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .confidence-interval {
        background: #e8f4fd;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .feature-impact {
        background: #fff3cd;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
    }
    .positive-impact {
        border-left: 4px solid #28a745;
    }
    .negative-impact {
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

def call_credit_score_api(customer_id, debug=True, nl_explanation=True):
    """Call the credit score API endpoint"""
    url = "http://localhost:5001/credit-score/customer-insights"
    
    payload = {
        "customer_id": customer_id,
        "debug": debug,
        "nl_explanation": nl_explanation
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error calling API: {str(e)}")
        return None

def display_credit_score(result):
    """Display the main credit score information"""
    if not result:
        return
    
    # Main score display
    predicted_score = result.get('predicted_credit_score', 0)
    actual_score = result.get('actual_credit_score')
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"""
        <div class="score-card">
            <h2>📊 Credit Score Prediction</h2>
            <h1 style="font-size: 3rem; margin: 1rem 0;">{predicted_score}</h1>
            <p style="font-size: 1.2rem;">Predicted Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Confidence intervals
    confidence_intervals = result.get('confidence_intervals', {})
    if confidence_intervals:
        st.subheader("🎯 Confidence Intervals")
        col1, col2 = st.columns(2)
        
        with col1:
            ci_68 = confidence_intervals.get('68_percent', {})
            if ci_68:
                st.markdown(f"""
                <div class="confidence-interval">
                    <strong>68% Confidence Interval:</strong><br>
                    {ci_68.get('lower', 'N/A')} - {ci_68.get('upper', 'N/A')}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            ci_95 = confidence_intervals.get('95_percent', {})
            if ci_95:
                st.markdown(f"""
                <div class="confidence-interval">
                    <strong>95% Confidence Interval:</strong><br>
                    {ci_95.get('lower', 'N/A')} - {ci_95.get('upper', 'N/A')}
                </div>
                """, unsafe_allow_html=True)

def display_explanations(result):
    """Display the natural language explanations"""
    explanations = result.get('explanations', {})
    if not explanations:
        return
    
    st.subheader("📝 Product-Specific Explanations")
    
    # Create tabs for different products
    product_names = {
        'auto-loan': '🚗 Auto Loan',
        'credit-card': '💳 Credit Card',
        'digital-savings': '🏦 Digital Savings',
        'home-loan': '🏠 Home Loan'
    }
    
    tabs = st.tabs([product_names.get(product, product.title()) for product in explanations.keys()])
    
    for i, (product, explanation) in enumerate(explanations.items()):
        with tabs[i]:
            # Check if explanation is an error message
            if isinstance(explanation, dict) and 'error' in explanation:
                error_msg = explanation.get('error', 'Unknown error')
                st.markdown(f"""
                <div class="explanation-card" style="border-left: 4px solid #ffc107; background: #fff3cd;">
                    <h4>⚠️ Account Not Available</h4>
                    <p style="color: #856404; margin: 0;">
                        <strong>Product:</strong> {product_names.get(product, product.title())}<br>
                        <strong>Status:</strong> {error_msg}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="explanation-card">
                    <h4>{product_names.get(product, product.title())}</h4>
                    <p>{explanation}</p>
                </div>
                """, unsafe_allow_html=True)

def display_feature_explanations(result, debug_mode=False):
    """Display feature-level explanations"""
    feature_explanations = result.get('feature_explanations', {})
    if not feature_explanations:
        return
    
    st.subheader("🔍 Feature Impact Analysis")
    
    # Create tabs for different products
    product_names = {
        'auto-loan': '🚗 Auto Loan',
        'credit-card': '💳 Credit Card',
        'digital-savings': '🏦 Digital Savings',
        'home-loan': '🏠 Home Loan'
    }
    
    tabs = st.tabs([product_names.get(product, product.title()) for product in feature_explanations.keys()])
    
    for i, (product, features) in enumerate(feature_explanations.items()):
        with tabs[i]:
            if features and isinstance(features, (list, tuple)):
                for feature in features:
                    try:
                        # Add debug logging only if debug mode is enabled
                        if debug_mode:
                            st.write(f"🔍 Debug - Feature structure: {feature}")
                        
                        # Handle different possible data structures
                        if isinstance(feature, dict):
                            direction = feature.get('direction', 'Unknown')
                            impact = feature.get('impact', 'Unknown')
                            feature_name = feature.get('feature_name', 'Unknown')
                        elif isinstance(feature, str):
                            # If feature is just a string, display it as is
                            st.markdown(f"""
                            <div class="feature-impact">
                                <strong>📊 {feature}</strong>
                            </div>
                            """, unsafe_allow_html=True)
                            continue
                        else:
                            # Handle other data types
                            st.markdown(f"""
                            <div class="feature-impact">
                                <strong>📊 Feature: {str(feature)}</strong>
                            </div>
                            """, unsafe_allow_html=True)
                            continue
                        
                        # Determine color based on direction
                        border_class = "positive-impact" if direction == "Positive" else "negative-impact"
                        direction_icon = "📈" if direction == "Positive" else "📉"
                        
                        st.markdown(f"""
                        <div class="feature-impact {border_class}">
                            <strong>{direction_icon} {feature_name}</strong><br>
                            <strong>Direction:</strong> {direction}<br>
                            <strong>Impact Level:</strong> {impact}
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"❌ Error processing feature: {str(e)}")
                        if debug_mode:
                            st.write(f"🔍 Raw feature data: {feature}")
            elif features:
                # Handle case where features is not a list/tuple
                if isinstance(features, dict) and 'error' in features:
                    # Handle API error messages
                    error_msg = features.get('error', 'Unknown error')
                    st.markdown(f"""
                    <div class="explanation-card" style="border-left: 4px solid #ffc107; background: #fff3cd;">
                        <h4>⚠️ Account Not Available</h4>
                        <p style="color: #856404; margin: 0;">
                            <strong>Product:</strong> {product_names.get(product, product.title())}<br>
                            <strong>Status:</strong> {error_msg}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Handle other unexpected data types
                    st.info(f"Features for {product} is not in expected format: {type(features)}")
                    if debug_mode:
                        st.write(f"Features data: {features}")
            else:
                st.info("No feature explanations available for this product.")

def display_debug_info(result):
    """Display debug information if available"""
    if 'actual_credit_score' in result or 'prediction_error' in result:
        st.subheader("🔧 Debug Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            actual_score = result.get('actual_credit_score')
            if actual_score:
                st.metric("Actual Credit Score", actual_score)
        
        with col2:
            prediction_error = result.get('prediction_error', {})
            if prediction_error:
                absolute_error = prediction_error.get('absolute_error', 0)
                percentage_error = prediction_error.get('percentage_error', 0)
                st.metric("Absolute Error", f"{absolute_error:.2f}")
                st.metric("Percentage Error", f"{percentage_error:.2f}%")

def main():
    # Header
    st.markdown('<h1 class="main-header">🏦 Credit Score Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar for input
    with st.sidebar:
        st.header("📋 Input Parameters")
        
        customer_id = st.text_input(
            "Customer ID",
            value="100-16-1590",
            help="Enter the customer's tax ID (e.g., 100-16-1590)"
        )
        
        debug_mode = st.checkbox(
            "Debug Mode",
            value=True,
            help="Enable to see actual credit score and prediction errors"
        )
        
        nl_explanation = st.checkbox(
            "Natural Language Explanations",
            value=True,
            help="Enable to get detailed explanations for each product"
        )
        
        predict_button = st.button(
            "🚀 Predict Credit Score",
            type="primary",
            use_container_width=True
        )
    
    # Main content area
    if predict_button and customer_id:
        with st.spinner("🔮 Predicting credit score..."):
            result = call_credit_score_api(customer_id, debug_mode, nl_explanation)
        
        if result:
            # Display timestamp
            timestamp = result.get('timestamp', '')
            if timestamp:
                st.caption(f"🕒 Analysis performed at: {timestamp}")
            
            # Display main results
            display_credit_score(result)
            
            # Display explanations
            if nl_explanation:
                display_explanations(result)
<<<<<<< HEAD
            
            # Display feature explanations (always show if available)
            display_feature_explanations(result, debug_mode)
=======
                display_feature_explanations(result, debug_mode)
>>>>>>> 49bd1b9136b4f5bf4d8e431592568d2cbfb26ad2
            
            # Display debug info if available
            if debug_mode:
                display_debug_info(result)
            
            # Show raw JSON in expander for debugging
            with st.expander("🔍 View Raw API Response"):
                st.json(result)
                
            # Additional debug information for feature explanations
<<<<<<< HEAD
            if 'feature_explanations' in result:
=======
            if nl_explanation and 'feature_explanations' in result:
>>>>>>> 49bd1b9136b4f5bf4d8e431592568d2cbfb26ad2
                with st.expander("🔍 Debug Feature Explanations Structure"):
                    st.write("Feature explanations keys:", list(result['feature_explanations'].keys()))
                    for product, features in result['feature_explanations'].items():
                        st.write(f"Product: {product}")
                        st.write(f"Features type: {type(features)}")
                        st.write(f"Features: {features}")
                        if features and isinstance(features, (list, tuple)) and len(features) > 0:
                            st.write(f"First feature type: {type(features[0])}")
                            st.write(f"First feature: {features[0]}")
                        elif features:
                            st.write(f"Features is not a list/tuple: {type(features)}")
                        else:
                            st.write("Features is empty or None")
        else:
            st.error("❌ Failed to get prediction results. Please check the API server is running.")
    
    elif predict_button and not customer_id:
        st.warning("⚠️ Please enter a Customer ID")
    
    # Instructions
    with st.expander("ℹ️ How to use this tool"):
        st.markdown("""
        ### Instructions:
        1. **Enter Customer ID**: Input the customer's tax ID (e.g., 100-16-1590)
        2. **Configure Options**: 
           - Enable Debug Mode to see actual scores and errors
           - Enable Natural Language Explanations for detailed insights
        3. **Click Predict**: Get comprehensive credit score analysis
        
        ### What you'll see:
        - **Predicted Credit Score**: The main prediction
        - **Confidence Intervals**: Range of possible scores
        - **Product Explanations**: Detailed insights for each financial product
        - **Feature Analysis**: Which factors most impact the score
        
        ### API Endpoint:
        This UI calls the `/credit-score/customer-insights` endpoint running on `localhost:5001`
        """)

if __name__ == "__main__":
    main() 