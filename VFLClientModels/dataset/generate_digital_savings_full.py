import pandas as pd
import numpy as np
import os

def determine_savings_category(row):
    """
    Determine customer category based on weighted factors using ALL available features:
    - savings_balance (20%): Higher balances indicate better category
    - digital_banking_score (15%): Level of digital channel usage
    - payment_history (12%): Better payment history indicates reliability
    - annual_income (12%): Higher income indicates better potential
    - credit_score (10%): Overall creditworthiness
    - investment_balance (8%): Investment portfolio size
    - checking_balance (6%): Checking account activity
    - avg_monthly_transactions (5%): Transaction frequency
    - mobile_banking_usage (4%): Mobile engagement
    - online_transactions_ratio (3%): Digital adoption
    - employment_length (2%): Job stability
    - age (2%): Customer maturity
    - e_statement_enrolled (1%): Digital adoption
    
    Categories:
    - Regular: Basic customers
    - Preferred: Mid-tier customers with good financial standing
    - VIP: Top-tier customers with excellent financial metrics
    """
    # Calculate individual component scores (0-100)
    savings_score = min(100, (row['savings_balance'] / 100000) * 100)
    digital_score = row['digital_banking_score']
    payment_score = row['payment_history']
    income_score = min(100, (row['annual_income'] / 150000) * 100)
    credit_score_norm = min(100, max(0, (row['credit_score'] - 300) / 550 * 100))
    investment_score = min(100, (row['investment_balance'] / 200000) * 100)
    checking_score = min(100, (row['checking_balance'] / 50000) * 100)
    transaction_score = min(100, row['avg_monthly_transactions'] * 2)
    mobile_score = row['mobile_banking_usage']
    online_score = row['online_transactions_ratio'] * 100
    employment_score = min(100, row['employment_length'] * 5)
    age_score = min(100, max(0, (row['age'] - 18) * 2))
    e_statement_score = row['e_statement_enrolled'] * 100
    
    # Calculate weighted total score
    total_score = (
        savings_score * 0.20 +
        digital_score * 0.15 +
        payment_score * 0.12 +
        income_score * 0.12 +
        credit_score_norm * 0.10 +
        investment_score * 0.08 +
        checking_score * 0.06 +
        transaction_score * 0.05 +
        mobile_score * 0.04 +
        online_score * 0.03 +
        employment_score * 0.02 +
        age_score * 0.02 +
        e_statement_score * 0.01
    )
    
    # Additional bonus points for high-value customers
    if (row['savings_balance'] >= 75000 and 
        row['payment_history'] >= 90 and 
        row['digital_banking_score'] >= 80):
        total_score += 10
    
    if (row['annual_income'] >= 100000 and 
        row['investment_balance'] >= 50000):
        total_score += 8
    
    if (row['credit_score'] >= 750 and
        row['credit_utilization_ratio'] <= 0.3):
        total_score += 5
    
    # Determine category based on thresholds
    if total_score >= 75:
        return 'VIP'
    elif total_score >= 55:
        return 'Preferred'
    else:
        return 'Regular'

def calculate_additional_digital_metrics(df):
    """Calculate additional digital engagement and financial metrics"""
    # Digital activity composite score
    df['digital_activity_score'] = (
        df['digital_banking_score'] * 0.4 +
        df['mobile_banking_usage'] * 0.3 +
        df['online_transactions_ratio'] * 100 * 0.3
    ).round(2)
    
    # Monthly digital transactions
    df['monthly_digital_transactions'] = (
        df['avg_monthly_transactions'] * df['online_transactions_ratio']
    ).round(1)
    
    # Total liquid assets
    df['total_liquid_assets'] = (
        df['savings_balance'] + df['checking_balance']
    ).round(2)
    
    # Total portfolio value
    df['total_portfolio_value'] = (
        df['savings_balance'] + df['checking_balance'] + df['investment_balance']
    ).round(2)
    
    # Digital engagement level
    conditions = [
        (df['digital_activity_score'] >= 80),
        (df['digital_activity_score'] >= 60),
        (df['digital_activity_score'] >= 40)
    ]
    choices = ['High', 'Medium', 'Low']
    df['digital_engagement_level'] = np.select(conditions, choices, default='Very Low')
    
    # Financial health score (0-100)
    df['financial_health_score'] = (
        # Income component (25%)
        (np.minimum(df['annual_income'] / 100000, 1) * 25) +
        # Savings component (25%)
        (np.minimum(df['savings_balance'] / 50000, 1) * 25) +
        # Credit component (20%)
        ((df['credit_score'] - 300) / 550 * 20) +
        # Payment history component (15%)
        (df['payment_history'] * 0.15) +
        # Debt management component (15%)
        ((1 - df['debt_to_income_ratio']) * 15)
    ).clip(0, 100).round(2)
    
    return df

def create_digital_savings_full_dataset():
    """Create comprehensive digital savings bank dataset using ALL rows and columns"""
    # Read the original dataset
    df = pd.read_csv('data/credit_scoring_dataset.csv')
    
    print(f"Original dataset: {len(df)} rows, {len(df.columns)} columns")
    print("Using ALL rows and ALL columns for digital savings dataset")
    
    # Use ALL rows (no sampling)
    savings_df = df.copy()
    
    # Calculate additional metrics using all available data
    savings_df = calculate_additional_digital_metrics(savings_df)
    
    # Add customer category based on comprehensive analysis
    savings_df['customer_category'] = savings_df.apply(determine_savings_category, axis=1)
    
    # Add risk assessment based on comprehensive data
    savings_df['risk_assessment'] = np.select([
        (savings_df['credit_score'] >= 750) & (savings_df['debt_to_income_ratio'] <= 0.3),
        (savings_df['credit_score'] >= 650) & (savings_df['debt_to_income_ratio'] <= 0.4),
        (savings_df['credit_score'] >= 550) & (savings_df['debt_to_income_ratio'] <= 0.5)
    ], ['Low Risk', 'Medium Risk', 'High Risk'], default='Very High Risk')
    
    # Add account type recommendation
    conditions = [
        (savings_df['total_portfolio_value'] >= 500000) & (savings_df['customer_category'] == 'VIP'),
        (savings_df['total_portfolio_value'] >= 100000) & (savings_df['customer_category'].isin(['VIP', 'Preferred'])),
        (savings_df['total_portfolio_value'] >= 25000)
    ]
    choices = ['Private Banking', 'Premium Savings', 'Standard Savings']
    savings_df['recommended_account_type'] = np.select(conditions, choices, default='Basic Savings')
    
    # Add investment readiness score
    savings_df['investment_readiness_score'] = (
        # Current investment activity (30%)
        (np.minimum(savings_df['investment_balance'] / 100000, 1) * 30) +
        # Financial stability (25%)
        (savings_df['financial_health_score'] * 0.25) +
        # Income level (20%)
        (np.minimum(savings_df['annual_income'] / 150000, 1) * 20) +
        # Digital engagement (15%)
        (savings_df['digital_activity_score'] * 0.15) +
        # Age factor (10%)
        (np.clip((savings_df['age'] - 25) / 40, 0, 1) * 10)
    ).clip(0, 100).round(2)
    
    # Print comprehensive dataset statistics
    print("\nDigital Savings Bank Dataset (Full) Statistics:")
    print(f"Total records: {len(savings_df)} (100% of original)")
    print(f"Total features: {len(savings_df.columns)}")
    
    print("\nOriginal Credit Data Features:")
    original_features = [col for col in df.columns]
    for i, feature in enumerate(original_features, 1):
        print(f"{i:2d}. {feature}")
    
    print("\nAdditional Computed Features:")
    new_features = [col for col in savings_df.columns if col not in df.columns]
    for i, feature in enumerate(new_features, 1):
        print(f"{i:2d}. {feature}")
    
    print(f"\nTotal features in final dataset: {len(savings_df.columns)}")
    
    print("\nCustomer Category Distribution:")
    category_dist = savings_df['customer_category'].value_counts()
    print(category_dist)
    print("\nCategory Percentages:")
    print((category_dist / len(savings_df) * 100).round(2))
    
    print("\nRisk Assessment Distribution:")
    risk_dist = savings_df['risk_assessment'].value_counts()
    print(risk_dist)
    
    print("\nRecommended Account Type Distribution:")
    account_dist = savings_df['recommended_account_type'].value_counts()
    print(account_dist)
    
    print("\nDigital Engagement Level Distribution:")
    digital_dist = savings_df['digital_engagement_level'].value_counts()
    print(digital_dist)
    
    print("\nKey Statistics by Customer Category:")
    for category in ['Regular', 'Preferred', 'VIP']:
        print(f"\n{category} Category:")
        category_data = savings_df[savings_df['customer_category'] == category]
        if len(category_data) > 0:
            print(f"  Count: {len(category_data)}")
            print(f"  Avg Credit Score: {category_data['credit_score'].mean():.2f}")
            print(f"  Avg Annual Income: ${category_data['annual_income'].mean():,.2f}")
            print(f"  Avg Savings Balance: ${category_data['savings_balance'].mean():,.2f}")
            print(f"  Avg Digital Score: {category_data['digital_banking_score'].mean():.2f}")
            print(f"  Avg Financial Health: {category_data['financial_health_score'].mean():.2f}")
            print(f"  Avg Investment Readiness: {category_data['investment_readiness_score'].mean():.2f}")
    
    # Save the comprehensive dataset
    output_path = 'data/banks/digital_savings_bank_full.csv'
    savings_df.to_csv(output_path, index=False)
    print(f"\nComprehensive dataset saved to: {output_path}")
    
    return savings_df

def main():
    """Generate comprehensive digital savings bank dataset"""
    # Create directories
    os.makedirs('data/banks', exist_ok=True)
    
    # Generate comprehensive digital savings dataset
    print("Generating comprehensive digital savings bank dataset...")
    print("Using ALL rows and ALL columns from credit scoring dataset")
    savings_df = create_digital_savings_full_dataset()
    
    print(f"\nDataset generation complete!")
    print(f"Final dataset: {len(savings_df)} rows, {len(savings_df.columns)} columns")

if __name__ == "__main__":
    main() 