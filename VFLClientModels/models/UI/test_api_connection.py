#!/usr/bin/env python3
"""
Test script to verify the credit score API is running and accessible.
Run this before starting the Streamlit UI to ensure the API is working.
"""

import requests
import json
import sys

def test_api_connection():
    """Test the credit score API endpoint"""
    
    print("🔍 Testing Credit Score API Connection...")
    print("=" * 50)
    
    # Test URL
    url = "http://localhost:5000/credit-score/customer-insights"
    
    # Test payload
    payload = {
        "customer_id": "100-16-1590",
        "debug": True,
        "nl_explanation": True
    }
    
    try:
        print(f"📡 Calling API at: {url}")
        print(f"📋 Request payload: {json.dumps(payload, indent=2)}")
        print()
        
        # Make the request
        response = requests.post(url, json=payload, timeout=30)
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ API is working correctly!")
            data = response.json()
            
            # Display key information
            print("\n📈 Key Results:")
            print(f"   Customer ID: {data.get('customer_id', 'N/A')}")
            print(f"   Predicted Score: {data.get('predicted_credit_score', 'N/A')}")
            print(f"   Actual Score: {data.get('actual_credit_score', 'N/A')}")
            
            # Check for explanations
            explanations = data.get('explanations', {})
            if explanations:
                print(f"   📝 Product Explanations: {len(explanations)} available")
                for product in explanations.keys():
                    print(f"      - {product}")
            
            # Check for feature explanations
            feature_explanations = data.get('feature_explanations', {})
            if feature_explanations:
                print(f"   🔍 Feature Explanations: {len(feature_explanations)} products")
                for product in feature_explanations.keys():
                    features = feature_explanations[product]
                    print(f"      - {product}: {len(features)} features")
            
            print("\n🎉 API test successful! You can now run the Streamlit UI.")
            return True
            
        else:
            print(f"❌ API returned error status: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error details: {error_data}")
            except:
                print(f"   Response text: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Could not connect to the API server.")
        print("   Make sure the API server is running on localhost:5000")
        print("   Run: cd VFLClientModels/models/apis && python app.py")
        return False
        
    except requests.exceptions.Timeout:
        print("❌ Timeout Error: The API request timed out.")
        print("   The server might be overloaded or not responding.")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False

def main():
    """Main function to run the API test"""
    print("🏦 Credit Score API Connection Test")
    print("=" * 50)
    
    success = test_api_connection()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed! Ready to run the Streamlit UI.")
        print("   Run: streamlit run credit_score_ui.py")
    else:
        print("❌ API test failed. Please fix the issues before running the UI.")
        sys.exit(1)

if __name__ == "__main__":
    main() 