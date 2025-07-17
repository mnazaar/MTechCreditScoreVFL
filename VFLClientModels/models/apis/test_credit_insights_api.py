#!/usr/bin/env python3
"""
Test script for the new /credit-score/customer-insights API endpoint

This script demonstrates how to use the new API endpoint with both debug=true and debug=false scenarios.

Usage:
    python test_credit_insights_api.py
"""

import requests
import json
import sys

def test_credit_insights_api():
    """Test the credit insights API endpoint"""
    
    # API base URL
    base_url = "http://localhost:5000"
    
    # Test customer ID (replace with actual customer ID from your dataset)
    test_customer_id = "100-13-3553"  # Example customer ID
    
    print("🚀 Testing Credit Score Customer Insights API")
    print("=" * 50)
    print(f"Base URL: {base_url}")
    print(f"Test Customer ID: {test_customer_id}")
    print()
    
    # Test 1: Production mode (debug=false)
    print("📊 Test 1: Production Mode (debug=false)")
    print("-" * 40)
    
    try:
        response = requests.post(
            f"{base_url}/credit-score/customer-insights",
            json={
                "customer_id": test_customer_id,
                "debug": False
            },
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Success! Response:")
            print(f"   Customer ID: {data.get('customer_id')}")
            print(f"   Predicted Credit Score: {data.get('predicted_credit_score')}")
            print(f"   Confidence Level: {data.get('confidence_level')}")
            print(f"   Confidence Score: {data.get('confidence_score')}")
            print(f"   Prediction Uncertainty: ±{data.get('prediction_uncertainty')} points")
            
            # Show confidence intervals
            ci_68 = data.get('confidence_intervals', {}).get('68_percent', {})
            ci_95 = data.get('confidence_intervals', {}).get('95_percent', {})
            print(f"   68% Confidence Interval: {ci_68.get('lower')} - {ci_68.get('upper')}")
            print(f"   95% Confidence Interval: {ci_95.get('lower')} - {ci_95.get('upper')}")
            
            # Show available services
            services = data.get('available_services', {})
            print(f"   Available Services:")
            for service, available in services.items():
                status = "✅" if available else "❌"
                print(f"     {status} {service.replace('_', ' ').title()}")
            
            # Verify debug info is NOT present
            if 'actual_credit_score' not in data and 'prediction_error' not in data:
                print("   ✅ Debug information correctly excluded")
            else:
                print("   ⚠️  Debug information unexpectedly present")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the API server is running on localhost:5000")
        print("   Start the server with: python app.py")
        return
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
    
    print()
    
    # Test 2: Debug mode (debug=true)
    print("🔍 Test 2: Debug Mode (debug=true)")
    print("-" * 40)
    
    try:
        response = requests.post(
            f"{base_url}/credit-score/customer-insights",
            json={
                "customer_id": test_customer_id,
                "debug": True
            },
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Success! Response:")
            print(f"   Customer ID: {data.get('customer_id')}")
            print(f"   Predicted Credit Score: {data.get('predicted_credit_score')}")
            print(f"   Confidence Level: {data.get('confidence_level')}")
            print(f"   Confidence Score: {data.get('confidence_score')}")
            print(f"   Prediction Uncertainty: ±{data.get('prediction_uncertainty')} points")
            
            # Show confidence intervals
            ci_68 = data.get('confidence_intervals', {}).get('68_percent', {})
            ci_95 = data.get('confidence_intervals', {}).get('95_percent', {})
            print(f"   68% Confidence Interval: {ci_68.get('lower')} - {ci_68.get('upper')}")
            print(f"   95% Confidence Interval: {ci_95.get('lower')} - {ci_95.get('upper')}")
            
            # Show available services
            services = data.get('available_services', {})
            print(f"   Available Services:")
            for service, available in services.items():
                status = "✅" if available else "❌"
                print(f"     {status} {service.replace('_', ' ').title()}")
            
            # Show debug information
            if 'actual_credit_score' in data:
                print(f"   🔍 DEBUG - Actual Credit Score: {data.get('actual_credit_score')}")
            
            if 'prediction_error' in data:
                error_info = data.get('prediction_error', {})
                print(f"   🔍 DEBUG - Prediction Error: {error_info.get('absolute_error')} points ({error_info.get('percentage_error')}%)")
            
            # Show model info
            model_info = data.get('model_info', {})
            if model_info:
                print(f"   🔍 DEBUG - Model Type: {model_info.get('model_type')}")
                print(f"   🔍 DEBUG - Representation Sizes: {model_info.get('representation_sizes')}")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
    
    print()
    
    # Test 3: Error handling - Invalid customer ID
    print("⚠️  Test 3: Error Handling (Invalid Customer ID)")
    print("-" * 40)
    
    try:
        response = requests.post(
            f"{base_url}/credit-score/customer-insights",
            json={
                "customer_id": "INVALID-CUSTOMER-ID",
                "debug": False
            },
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code != 200:
            data = response.json()
            print(f"Error Response: {data.get('error', 'Unknown error')}")
        else:
            print("Unexpected success with invalid customer ID")
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
    
    print()
    print("🎉 API Testing Complete!")
    print("=" * 50)

def test_health_check():
    """Test the health check endpoint"""
    print("🏥 Testing Health Check")
    print("-" * 30)
    
    try:
        response = requests.get("http://localhost:5000/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ Health Check Passed!")
            print(f"   Status: {data.get('status')}")
            print(f"   Timestamp: {data.get('timestamp')}")
            print(f"   Auto Loans Models: {'✅' if data.get('auto_loans_models_loaded') else '❌'}")
            print(f"   Home Loans Models: {'✅' if data.get('home_loans_models_loaded') else '❌'}")
            print(f"   Credit Card Models: {'✅' if data.get('credit_card_models_loaded') else '❌'}")
            print(f"   Digital Savings Models: {'✅' if data.get('digital_savings_models_loaded') else '❌'}")
        else:
            print(f"❌ Health Check Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health Check Error: {str(e)}")

if __name__ == "__main__":
    print("🧪 Credit Score Customer Insights API Test Suite")
    print("=" * 60)
    
    # First test health check
    test_health_check()
    print()
    
    # Then test the main functionality
    test_credit_insights_api()
    
    print("\n💡 Usage Examples:")
    print("   # Production mode (no debug info)")
    print("   curl -X POST http://localhost:5000/credit-score/customer-insights \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{\"customer_id\": \"100-13-3553\", \"debug\": false}'")
    print()
    print("   # Debug mode (with all details)")
    print("   curl -X POST http://localhost:5000/credit-score/customer-insights \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{\"customer_id\": \"100-13-3553\", \"debug\": true}'") 