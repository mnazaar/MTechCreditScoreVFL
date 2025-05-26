import pandas as pd
import numpy as np
import os

def calculate_auto_loan_limit(row):
    """
    Calculate auto loan limit based on standalone criteria
    Rules:
    1. Base Qualification:
       - Credit score must be > 650 (increased from 580)
       - Payment history must be > 0.70 (increased from 0.60)
       - Employment length >= 1 year (increased from 0.5)
       - Age >= 21 (unchanged)
       - Debt-to-income ratio must be < 0.50 (reduced from 0.60)
       - Credit utilization must be < 0.75 (reduced from 0.85)
       
    2. Loan Multiplier based on credit score:
       - 4.0x income if credit score > 750 (Excellent)
       - 3.5x income if credit score > 700 (Very Good)
       - 3.0x income if credit score > 675 (Good)
       - 2.5x income if credit score > 650 (Fair)
       
    3. Adjustments:
       - Reduce by 30% if debt_to_income_ratio > 0.40 (stricter threshold)
       - Reduce by 25% if credit_utilization_ratio > 0.60 (stricter threshold)
       - Reduce by 15% for each existing loan account (increased from 10%)
       - Add 10% if employment_length > 5 years (unchanged)
    """
    # Extract features
    annual_income = row['annual_income']
    credit_score = row['credit_score']
    payment_history = row['payment_history']
    employment_length = row['employment_length']
    debt_ratio = row['debt_to_income_ratio']
    util_ratio = row['credit_utilization_ratio']
    num_loans = row['num_loan_accounts']
    age = row['age']
    
    # Basic eligibility checks
    if (credit_score <= 650 or 
        payment_history <= 0.70 or 
        employment_length < 1 or 
        age < 21 or
        debt_ratio >= 0.50 or
        util_ratio >= 0.75):
        return 0
    
    # Base multiplier based on credit score
    if credit_score > 750:
        multiplier = 4.0
    elif credit_score > 700:
        multiplier = 3.5
    elif credit_score > 675:
        multiplier = 3.0
    else:  # > 650
        multiplier = 2.5
    
    # Calculate base loan amount
    loan_limit = annual_income * multiplier
    
    # Apply adjustments
    if debt_ratio > 0.40:
        loan_limit *= 0.70  # Reduce by 30%
    if util_ratio > 0.60:
        loan_limit *= 0.75  # Reduce by 25%
    
    # Reduce for existing loans
    loan_reduction = max(0, 1 - (num_loans * 0.15))  # Cap reduction at 100%
    loan_limit *= loan_reduction
    
    # Bonus for long employment (unchanged)
    if employment_length > 5:
        loan_limit *= 1.1  # Add 10%
    
    # Apply minimum loan amount rule
    if loan_limit < 5000:  # Increased minimum loan amount
        return 0
    
    # Round to nearest thousand
    return round(loan_limit / 1000) * 1000

def generate_auto_loans_dataset():
    """Generate auto loans dataset from credit scoring data"""
    # Create output directory if it doesn't exist
    os.makedirs('VFLClientModels/dataset/data/banks', exist_ok=True)
    
    # Read credit scoring dataset
    credit_df = pd.read_csv('VFLClientModels/dataset/data/credit_scoring_dataset.csv')
    
    # Randomly select 80% of records with different seed for auto loans
    np.random.seed(123)  # Different seed from digital loans
    selected_indices = np.random.random(len(credit_df)) <= 0.8
    credit_df = credit_df[selected_indices].copy()
    
    print("\nCredit Score Distribution:")
    print(credit_df['credit_score'].describe())
    
    print("\nPayment History Distribution:")
    print(credit_df['payment_history'].describe())
    
    print("\nEmployment Length Distribution:")
    print(credit_df['employment_length'].describe())
    
    print("\nDebt Ratio Distribution:")
    print(credit_df['debt_to_income_ratio'].describe())
    
    print("\nCredit Utilization Distribution:")
    print(credit_df['credit_utilization_ratio'].describe())
    
    # Select relevant features for auto loans
    auto_loans_features = [
        'tax_id',                  # Customer identifier
        'annual_income',           # Income for loan calculation
        'credit_score',            # Credit worthiness
        'payment_history',         # Payment reliability
        'employment_length',       # Job stability
        'debt_to_income_ratio',    # Existing debt burden
        'age',                     # Age of customer
        'num_credit_cards',        # Existing credit relationships
        'num_loan_accounts',       # Existing loan burden
        'total_credit_limit',      # Overall credit exposure
        'credit_utilization_ratio' # Credit usage pattern
    ]
    
    # Create auto loans dataset
    auto_loans_df = credit_df[auto_loans_features].copy()
    
    # Calculate auto loan limits
    auto_loans_df['auto_loan_limit'] = auto_loans_df.apply(calculate_auto_loan_limit, axis=1)
    
    # Add loan eligibility flag
    auto_loans_df['loan_eligible'] = auto_loans_df['auto_loan_limit'] > 0
    
    # Calculate risk score (0-100)
    auto_loans_df['risk_score'] = (
        (auto_loans_df['credit_score'] - 300) / 550 * 35 +  # Credit score (35%)
        (auto_loans_df['payment_history']) * 0.20 +          # Payment history (20%)
        ((1 - auto_loans_df['debt_to_income_ratio']) * 100) * 0.15 +  # Debt ratio (15%)
        ((1 - auto_loans_df['credit_utilization_ratio']) * 100) * 0.10 + # Credit utilization (10%)
        (np.minimum(auto_loans_df['employment_length'], 10) * 10) * 0.10 + # Employment stability (10%)
        ((auto_loans_df['annual_income'] / 120000).clip(0, 1) * 100) * 0.05 + # Income level (5%)
        ((5 - auto_loans_df['num_loan_accounts']).clip(0, 5) * 20) * 0.05  # Existing loans (5%)
    ).clip(0, 100)
    
    # Add risk category
    conditions = [
        (auto_loans_df['risk_score'] >= 80),
        (auto_loans_df['risk_score'] >= 65),
        (auto_loans_df['risk_score'] >= 50),
        (auto_loans_df['risk_score'] >= 35)
    ]
    choices = ['Very Low Risk', 'Low Risk', 'Moderate Risk', 'High Risk']
    auto_loans_df['risk_category'] = np.select(conditions, choices, default='Very High Risk')
    
    # Add loan term eligibility (in months)
    conditions = [
        (auto_loans_df['risk_score'] >= 80),  # Excellent risk score
        (auto_loans_df['risk_score'] >= 65),  # Good risk score
        (auto_loans_df['risk_score'] >= 50),  # Fair risk score
        (auto_loans_df['risk_score'] >= 35)   # Poor risk score
    ]
    choices = [84, 72, 60, 48]  # 7, 6, 5, 4 years
    auto_loans_df['max_loan_term'] = np.select(conditions, choices, default=36)
    
    # Add interest rate based on risk score
    # Base rate of 5.99% for excellent scores
    # Each risk tier adds additional premium
    base_rate = 5.99
    risk_premium = (100 - auto_loans_df['risk_score']) * 0.15
    auto_loans_df['interest_rate'] = (base_rate + risk_premium).clip(5.99, 24.99)
    
    # Save the dataset
    output_path = 'VFLClientModels/dataset/data/banks/auto_loans_bank.csv'
    auto_loans_df.to_csv(output_path, index=False)
    print(f"Auto loans dataset saved to {output_path}")
    
    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Total records: {len(auto_loans_df)}")
    print(f"Eligible customers: {auto_loans_df['loan_eligible'].sum()}")
    
    print("\nRisk Category Distribution:")
    risk_dist = auto_loans_df['risk_category'].value_counts()
    print(risk_dist)
    print("\nRisk Category Percentages:")
    print((risk_dist / len(auto_loans_df) * 100).round(2), "%")
    
    print("\nAverage Interest Rate by Risk Category:")
    print(auto_loans_df.groupby('risk_category')['interest_rate'].mean().round(2))
    
    print("\nLoan Term Distribution:")
    print(auto_loans_df['max_loan_term'].value_counts().sort_index())
    
    print("\nAverage loan limits by credit score range:")
    credit_ranges = [
        (750, 850, 'Excellent (750+)'),
        (700, 750, 'Very Good (700-750)'),
        (650, 700, 'Good (650-700)'),
        (600, 650, 'Fair (600-650)'),
        (550, 600, 'Poor (550-600)'),
        (500, 550, 'Very Poor (500-550)'),
        (450, 500, 'Marginal (450-500)'),
        (300, 450, 'Not Eligible (<450)')
    ]
    for min_score, max_score, label in credit_ranges:
        mask = (auto_loans_df['credit_score'] > min_score) & (auto_loans_df['credit_score'] <= max_score)
        avg_limit = auto_loans_df.loc[mask, 'auto_loan_limit'].mean()
        count = mask.sum()
        print(f"\n{label}:")
        print(f"Count: {count}")
        print(f"Average Loan Limit: ${avg_limit:,.2f}")
        if count > 0:
            avg_risk = auto_loans_df.loc[mask, 'risk_score'].mean()
            avg_rate = auto_loans_df.loc[mask, 'interest_rate'].mean()
            print(f"Average Risk Score: {avg_risk:.2f}")
            print(f"Average Interest Rate: {avg_rate:.2f}%")

if __name__ == "__main__":
    generate_auto_loans_dataset() 