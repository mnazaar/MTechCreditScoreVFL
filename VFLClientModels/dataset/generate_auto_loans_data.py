import pandas as pd
import numpy as np
import os

def calculate_auto_loan_limit(row):
    """
    Calculate auto loan limit based on standalone criteria
    Returns both loan limit and eligibility status
    
    Eligibility Rules (RELAXED for ~80% approval):
    1. Base Qualification:
       - Credit score must be > 450 (reduced from 500)
       - Payment history must be > 0.30 (reduced from 0.40)
       - Employment length >= 0.25 year (reduced from 0.5)
       - Age >= 18 (unchanged)
       - Debt-to-income ratio must be < 0.90 (increased from 0.80)
       - Credit utilization must be < 0.95 (increased from 0.90)
       
    2. Loan Multiplier based on credit score:
       - 4.0x income if credit score > 750 (Excellent)
       - 3.5x income if credit score > 700 (Very Good)
       - 3.0x income if credit score > 675 (Good)
       - 2.5x income if credit score > 650 (Fair)
       - 2.0x income if credit score > 600 (Poor but gets some limit)
       - 1.5x income if credit score > 550 (Very Poor)
       - 1.0x income if credit score <= 550 (Minimal)
       
    3. Adjustments (applied to all customers):
       - Reduce by 20% if debt_to_income_ratio > 0.60 (more lenient)
       - Reduce by 15% if credit_utilization_ratio > 0.70 (more lenient)
       - Reduce by 10% for each existing loan account (more lenient)
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
    
    # Much more lenient eligibility criteria for ~80% approval
    is_eligible = (
        credit_score > 450 and  # Even lower credit score requirement
        payment_history > 30.0 and  # Even lower payment history requirement (30%)
        employment_length >= 0.25 and  # Even lower employment requirement (3 months)
        age >= 18 and  # Keep age requirement
        debt_ratio < 0.90 and  # Even higher DTI tolerance
        util_ratio < 0.95  # Even higher utilization tolerance
    )
    
    # Calculate loan limit for ALL customers using the same formula
    # Base multiplier based on credit score (extended to cover all ranges)
    if credit_score > 750:
        multiplier = 4.0
    elif credit_score > 700:
        multiplier = 3.5
    elif credit_score > 675:
        multiplier = 3.0
    elif credit_score > 650:
        multiplier = 2.5
    elif credit_score > 600:
        multiplier = 2.0
    elif credit_score > 550:
        multiplier = 1.5
    else:  # <= 550
        multiplier = 1.0
    
    # Calculate base loan amount
    loan_limit = annual_income * multiplier
    
    # Apply more lenient adjustments (same for all customers)
    if debt_ratio > 0.60:  # More lenient threshold
        loan_limit *= 0.80  # Reduce by 20% (less severe)
    if util_ratio > 0.70:  # More lenient threshold
        loan_limit *= 0.85  # Reduce by 15% (less severe)
    
    # Reduce for existing loans (more lenient)
    loan_reduction = max(0.2, 1 - (num_loans * 0.10))  # Minimum 20% of original, less severe reduction
    loan_limit *= loan_reduction
    
    # Bonus for long employment
    if employment_length > 5:
        loan_limit *= 1.1  # Add 10%
    
    # Apply minimum loan amount (but don't set to 0)
    loan_limit = max(1000, loan_limit)  # Minimum $1,000 loan limit
    
    # Round to nearest thousand
    loan_limit = round(loan_limit / 1000) * 1000
    
    return loan_limit, is_eligible

def generate_auto_loans_dataset():
    """Generate auto loans dataset from credit scoring data"""
    # Create output directory if it doesn't exist
    os.makedirs('data/banks', exist_ok=True)
    
    # Read credit scoring dataset
    credit_df = pd.read_csv('data/credit_scoring_dataset.csv')
    
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
        'tax_id',                       # Customer identifier
        'annual_income',                # Income for loan calculation
        'credit_score',                 # Credit worthiness
        'payment_history',              # Payment reliability
        'employment_length',            # Job stability
        'debt_to_income_ratio',         # Existing debt burden
        'age',                          # Age of customer
        'credit_history_length',        # Length of credit history
        'num_credit_cards',             # Existing credit relationships
        'num_loan_accounts',            # Existing loan burden
        'total_credit_limit',           # Overall credit exposure
        'credit_utilization_ratio',     # Credit usage pattern
        'late_payments',                # Payment behavior
        'credit_inquiries',             # Recent credit activity
        'current_debt',                 # Current debt amount
        'monthly_expenses',             # Monthly spending pattern
        'auto_loan_balance',            # Existing auto loan
        'mortgage_balance',             # Existing mortgage
        'investment_balance',           # Investment assets
        'last_late_payment_days',       # Recent payment behavior
        'savings_balance',              # Available savings
        'checking_balance'              # Liquid assets
    ]
    
    # Create auto loans dataset
    auto_loans_df = credit_df[auto_loans_features].copy()
    
    # Calculate auto loan limits and eligibility
    results = auto_loans_df.apply(calculate_auto_loan_limit, axis=1)
    auto_loans_df['auto_loan_limit'] = [result[0] for result in results]
    auto_loans_df['loan_eligible'] = [result[1] for result in results]
    
    # Calculate risk score (0-100) - Enhanced with new features
    auto_loans_df['risk_score'] = (
        (auto_loans_df['credit_score'] - 300) / 550 * 25 +  # Credit score (25%)
        (auto_loans_df['payment_history']) * 0.15 +          # Payment history (15%)
        ((1 - auto_loans_df['debt_to_income_ratio']) * 100) * 0.12 +  # Debt ratio (12%)
        ((1 - auto_loans_df['credit_utilization_ratio']) * 100) * 0.08 + # Credit utilization (8%)
        (np.minimum(auto_loans_df['employment_length'], 10) * 10) * 0.08 + # Employment stability (8%)
        ((auto_loans_df['annual_income'] / 120000).clip(0, 1) * 100) * 0.05 + # Income level (5%)
        ((5 - auto_loans_df['num_loan_accounts']).clip(0, 5) * 20) * 0.04 + # Existing loans (4%)
        (np.minimum(auto_loans_df['credit_history_length'], 30) / 30 * 100) * 0.06 + # Credit history length (6%)
        ((12 - auto_loans_df['late_payments']).clip(0, 12) / 12 * 100) * 0.05 + # Late payments (5%)
        ((10 - auto_loans_df['credit_inquiries']).clip(0, 10) / 10 * 100) * 0.03 + # Credit inquiries (3%)
        ((auto_loans_df['savings_balance'] + auto_loans_df['checking_balance']) / 50000).clip(0, 1) * 100 * 0.04 + # Liquid assets (4%)
        ((100000 - auto_loans_df['auto_loan_balance']).clip(0, 100000) / 100000 * 100) * 0.03 + # Existing auto loan burden (3%)
        ((auto_loans_df['investment_balance'] / 25000).clip(0, 1) * 100) * 0.02  # Investment assets (2%)
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
    output_path = 'data/banks/auto_loans_bank.csv'
    auto_loans_df.to_csv(output_path, index=False)
    print(f"Auto loans dataset saved to {output_path}")
    
    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Total records: {len(auto_loans_df)}")
    print(f"Eligible customers: {auto_loans_df['loan_eligible'].sum()}")
    print(f"Ineligible customers: {(~auto_loans_df['loan_eligible']).sum()}")
    print(f"Eligibility rate: {(auto_loans_df['loan_eligible'].sum() / len(auto_loans_df) * 100):.2f}%")
    
    print("\nLoan Limit Statistics:")
    print("All Customers:")
    print(auto_loans_df['auto_loan_limit'].describe())
    print("\nEligible Customers Only:")
    eligible_df = auto_loans_df[auto_loans_df['loan_eligible']]
    print(eligible_df['auto_loan_limit'].describe())
    print("\nIneligible Customers Only:")
    ineligible_df = auto_loans_df[~auto_loans_df['loan_eligible']]
    print(ineligible_df['auto_loan_limit'].describe())
    
    print("\nRisk Category Distribution:")
    risk_dist = auto_loans_df['risk_category'].value_counts()
    print(risk_dist)
    print("\nRisk Category Percentages:")
    print((risk_dist / len(auto_loans_df) * 100).round(2), "%")
    
    print("\nAverage Interest Rate by Risk Category:")
    print(auto_loans_df.groupby('risk_category')['interest_rate'].mean().round(2))
    
    print("\nLoan Term Distribution:")
    print(auto_loans_df['max_loan_term'].value_counts().sort_index())
    
    print("\nLoan statistics by credit score range:")
    credit_ranges = [
        (750, 850, 'Excellent (750+)'),
        (700, 750, 'Very Good (700-750)'),
        (650, 700, 'Good (650-700)'),
        (600, 650, 'Fair (600-650)'),
        (550, 600, 'Poor (550-600)'),
        (500, 550, 'Very Poor (500-550)'),
        (450, 500, 'Marginal (450-500)'),
        (300, 450, 'Very Low (<450)')
    ]
    
    for min_score, max_score, label in credit_ranges:
        mask = (auto_loans_df['credit_score'] > min_score) & (auto_loans_df['credit_score'] <= max_score)
        if mask.sum() > 0:
            subset = auto_loans_df.loc[mask]
            eligible_count = subset['loan_eligible'].sum()
            ineligible_count = (~subset['loan_eligible']).sum()
            
            print(f"\n{label}:")
            print(f"  Total Count: {mask.sum()}")
            print(f"  Eligible: {eligible_count} ({eligible_count/len(subset)*100:.1f}%)")
            print(f"  Ineligible: {ineligible_count} ({ineligible_count/len(subset)*100:.1f}%)")
            print(f"  Average Loan Limit (All): ${subset['auto_loan_limit'].mean():,.2f}")
            
            if eligible_count > 0:
                eligible_subset = subset[subset['loan_eligible']]
                print(f"  Average Loan Limit (Eligible): ${eligible_subset['auto_loan_limit'].mean():,.2f}")
                print(f"  Average Risk Score (Eligible): {eligible_subset['risk_score'].mean():.2f}")
                print(f"  Average Interest Rate (Eligible): {eligible_subset['interest_rate'].mean():.2f}%")
            
            if ineligible_count > 0:
                ineligible_subset = subset[~subset['loan_eligible']]
                print(f"  Average Loan Limit (Ineligible): ${ineligible_subset['auto_loan_limit'].mean():,.2f}")
                print(f"  Average Risk Score (Ineligible): {ineligible_subset['risk_score'].mean():.2f}")
                print(f"  Average Interest Rate (Ineligible): {ineligible_subset['interest_rate'].mean():.2f}%")

if __name__ == "__main__":
    generate_auto_loans_dataset() 