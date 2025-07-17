import pandas as pd
import numpy as np
import os

def calculate_auto_loan_limit(row):
    """
    Calculate auto loan limit based on comprehensive criteria using ALL available features
    Returns both loan limit and eligibility status
    
    Eligibility Rules (much more lenient):
    1. Base Qualification:
       - Credit score must be > 450 (was 650)
       - Payment history must be > 0.30 (was 0.70)
       - Employment length >= 0.5 years (was 1 year)
       - Age >= 18 (was 21)
       - Debt-to-income ratio must be < 0.85 (was 0.50)
       - Credit utilization must be < 0.95 (was 0.75)
       
    2. Loan Multiplier based on comprehensive scoring:
       - 5.0x income if comprehensive score > 90 (Excellent)
       - 4.5x income if comprehensive score > 80 (Very Good)
       - 4.0x income if comprehensive score > 70 (Good)
       - 3.5x income if comprehensive score > 60 (Fair)
       - 3.0x income if comprehensive score > 50 (Acceptable)
       - 2.5x income if comprehensive score > 40 (Marginal)
       - 2.0x income if comprehensive score > 30 (Poor)
       - 1.5x income if comprehensive score > 20 (Very Poor)
       - 1.0x income if comprehensive score <= 20 (Minimal)
       
    3. Comprehensive adjustments using all available data
    """
    # Extract all relevant features
    annual_income = row['annual_income']
    credit_score = row['credit_score']
    payment_history = row['payment_history']
    employment_length = row['employment_length']
    debt_ratio = row['debt_to_income_ratio']
    util_ratio = row['credit_utilization_ratio']
    num_loans = row['num_loan_accounts']
    age = row['age']
    savings_balance = row['savings_balance']
    checking_balance = row['checking_balance']
    investment_balance = row['investment_balance']
    credit_history_length = row['credit_history_length']
    late_payments = row['late_payments']
    credit_inquiries = row['credit_inquiries']
    num_credit_cards = row['num_credit_cards']
    
    # Check eligibility criteria (relaxed for ~80% approval rate)
    is_eligible = (
        credit_score > 450 and  # Only exclude very poor credit
        payment_history > 30.0 and  # Very lenient payment history (30%)
        employment_length >= 0.5 and  # 6 months minimum employment
        age >= 18 and  # Legal age only
        debt_ratio < 0.85 and  # Very high DTI threshold
        util_ratio < 0.95  # Almost maxed out utilization
    )

    # Calculate comprehensive creditworthiness score (0-100) for ALL customers
    comprehensive_score = (
        # Credit score component (25%)
        ((credit_score - 300) / 550 * 25) +
        # Payment history component (20%)
        (payment_history * 0.20) +
        # Financial stability component (15%)
        (min(100, (savings_balance + checking_balance) / 50000) * 15) +
        # Employment stability component (10%)
        (min(100, employment_length * 5) * 0.10) +
        # Credit history length component (10%)
        (min(100, credit_history_length * 5) * 0.10) +
        # Debt management component (10%)
        ((1 - debt_ratio) * 10) +
        # Credit utilization component (5%)
        ((1 - util_ratio) * 5) +
        # Age/maturity component (3%)
        (min(100, (age - 18) * 2) * 0.03) +
        # Credit portfolio diversity (2%)
        (min(100, num_credit_cards * 10) * 0.02)
    )
    
    # Reduced penalty for negative factors (more lenient)
    if late_payments > 5:  # Only penalize after 5 late payments
        comprehensive_score -= (late_payments - 5) * 2  # Reduced penalty
    if credit_inquiries > 6:  # More lenient on inquiries
        comprehensive_score -= (credit_inquiries - 6) * 1  # Reduced penalty
    if num_loans > 5:  # More lenient on existing loans
        comprehensive_score -= (num_loans - 5) * 2  # Reduced penalty
    
    # Ensure score is within bounds
    comprehensive_score = max(0, min(100, comprehensive_score))
    
    # Determine multiplier based on comprehensive score (extended for all ranges)
    if comprehensive_score > 90:
        multiplier = 5.0
    elif comprehensive_score > 80:
        multiplier = 4.5
    elif comprehensive_score > 70:
        multiplier = 4.0
    elif comprehensive_score > 60:
        multiplier = 3.5
    elif comprehensive_score > 50:
        multiplier = 3.0
    elif comprehensive_score > 40:
        multiplier = 2.5
    elif comprehensive_score > 30:
        multiplier = 2.0
    elif comprehensive_score > 20:
        multiplier = 1.5
    else:
        multiplier = 1.0  # Minimum for very poor credit

    # Calculate base loan amount for ALL customers
    loan_limit = annual_income * multiplier
    
    # Apply comprehensive adjustments (same for all customers)
    
    # Positive adjustments
    if savings_balance >= 100000:  # Strong savings
        loan_limit *= 1.15
    elif savings_balance >= 50000:
        loan_limit *= 1.10
    elif savings_balance >= 25000:
        loan_limit *= 1.05
    
    if investment_balance >= 100000:  # Investment portfolio
        loan_limit *= 1.10
    elif investment_balance >= 50000:
        loan_limit *= 1.05
    
    if employment_length > 10:  # Very stable employment
        loan_limit *= 1.15
    elif employment_length > 5:
        loan_limit *= 1.10
    
    if credit_history_length > 15:  # Long credit history
        loan_limit *= 1.08
    elif credit_history_length > 10:
        loan_limit *= 1.05
    
    # Reduced negative adjustments (more lenient)
    if debt_ratio > 0.70:  # Only penalize very high DTI
        loan_limit *= 0.85  # Less severe penalty
    if util_ratio > 0.80:  # Only penalize very high utilization
        loan_limit *= 0.90  # Less severe penalty
    if late_payments > 3:  # More lenient on late payments
        loan_limit *= (1 - ((late_payments - 3) * 0.03))  # Reduced penalty
    if credit_inquiries > 4:  # More lenient on inquiries
        loan_limit *= (1 - ((credit_inquiries - 4) * 0.02))  # Reduced penalty
    
    # Reduced penalty for existing loans
    loan_reduction = max(0.1, 1 - (num_loans * 0.08))  # Minimum 10% of original
    loan_limit *= loan_reduction
    
    # Apply minimum loan amount (but don't set to 0)
    loan_limit = max(1000, loan_limit)  # Minimum $1,000 loan limit
    
    # Round to nearest thousand
    loan_limit = round(loan_limit / 1000) * 1000
    
    return loan_limit, is_eligible

def calculate_comprehensive_auto_metrics(df):
    """Calculate additional auto loan metrics using all available data"""
    
    # Vehicle affordability index (0-100)
    df['vehicle_affordability_index'] = (
        # Income component (40%)
        (np.minimum(df['annual_income'] / 80000, 1) * 40) +
        # Savings component (25%)
        (np.minimum(df['savings_balance'] / 30000, 1) * 25) +
        # Credit component (20%)
        ((df['credit_score'] - 300) / 550 * 20) +
        # Debt management (15%)
        ((1 - df['debt_to_income_ratio']) * 15)
    ).clip(0, 100).round(2)
    
    # Loan-to-income ratio
    df['loan_to_income_ratio'] = (df['auto_loan_limit'] / df['annual_income']).round(4)
    
    # Total debt service ratio (including potential auto loan)
    estimated_auto_payment = df['auto_loan_limit'] * 0.02  # Assume 2% monthly payment
    df['total_debt_service_ratio'] = (
        (df['monthly_expenses'] + estimated_auto_payment) / (df['annual_income'] / 12)
    ).round(4)
    
    # Financial cushion score
    liquid_assets = df['savings_balance'] + df['checking_balance']
    monthly_income = df['annual_income'] / 12
    df['financial_cushion_months'] = (liquid_assets / monthly_income).round(1)
    
    # Credit portfolio strength
    df['credit_portfolio_strength'] = (
        # Credit history length (30%)
        (np.minimum(df['credit_history_length'] / 15, 1) * 30) +
        # Payment history (25%)
        (df['payment_history'] * 0.25) +
        # Credit utilization (20%)
        ((1 - df['credit_utilization_ratio']) * 20) +
        # Credit mix (15%)
        (np.minimum(df['num_credit_cards'] / 5, 1) * 15) +
        # Recent inquiries (10% - penalty)
        (np.maximum(0, 1 - df['credit_inquiries'] / 5) * 10)
    ).clip(0, 100).round(2)
    
    # Employment stability score
    df['employment_stability_score'] = (
        np.minimum(df['employment_length'] / 10, 1) * 100
    ).round(2)
    
    # Overall auto loan readiness score
    df['auto_loan_readiness_score'] = (
        # Credit score (25%)
        ((df['credit_score'] - 300) / 550 * 25) +
        # Income stability (20%)
        (df['employment_stability_score'] * 0.20) +
        # Financial cushion (20%)
        (np.minimum(df['financial_cushion_months'] / 6, 1) * 20) +
        # Credit portfolio (15%)
        (df['credit_portfolio_strength'] * 0.15) +
        # Debt management (10%)
        ((1 - df['debt_to_income_ratio']) * 10) +
        # Vehicle affordability (10%)
        (df['vehicle_affordability_index'] * 0.10)
    ).clip(0, 100).round(2)
    
    return df

def generate_auto_loans_full_dataset():
    """Generate comprehensive auto loans dataset using ALL rows and columns"""
    # Create output directory if it doesn't exist
    os.makedirs('data/banks', exist_ok=True)
    
    # Read credit scoring dataset
    credit_df = pd.read_csv('data/credit_scoring_dataset.csv')
    
    print(f"Original dataset: {len(credit_df)} rows, {len(credit_df.columns)} columns")
    print("Using ALL rows and ALL columns for auto loans dataset")
    
    # Use ALL rows (no sampling)
    auto_loans_df = credit_df.copy()
    
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
    
    # Calculate auto loan limits and eligibility using comprehensive analysis
    results = auto_loans_df.apply(calculate_auto_loan_limit, axis=1)
    auto_loans_df['auto_loan_limit'] = [result[0] for result in results]
    auto_loans_df['loan_eligible'] = [result[1] for result in results]
    
    # Calculate comprehensive metrics
    auto_loans_df = calculate_comprehensive_auto_metrics(auto_loans_df)
    
    # Enhanced risk score using all available data
    auto_loans_df['comprehensive_risk_score'] = (
        # Credit score (20%)
        ((auto_loans_df['credit_score'] - 300) / 550 * 20) +
        # Payment history (15%)
        (auto_loans_df['payment_history'] * 0.15) +
        # Employment stability (15%)
        (auto_loans_df['employment_stability_score'] * 0.15) +
        # Financial cushion (12%)
        (np.minimum(auto_loans_df['financial_cushion_months'] / 6, 1) * 12) +
        # Debt management (10%)
        ((1 - auto_loans_df['debt_to_income_ratio']) * 10) +
        # Credit utilization (8%)
        ((1 - auto_loans_df['credit_utilization_ratio']) * 8) +
        # Credit history (8%)
        (np.minimum(auto_loans_df['credit_history_length'] / 15, 1) * 8) +
        # Income level (7%)
        (np.minimum(auto_loans_df['annual_income'] / 120000, 1) * 7) +
        # Investment portfolio (3%)
        (np.minimum(auto_loans_df['investment_balance'] / 100000, 1) * 3) +
        # Age factor (2%)
        (np.minimum((auto_loans_df['age'] - 18) / 50, 1) * 2)
    ).clip(0, 100).round(2)
    
    # Enhanced risk category
    conditions = [
        (auto_loans_df['comprehensive_risk_score'] >= 85),
        (auto_loans_df['comprehensive_risk_score'] >= 70),
        (auto_loans_df['comprehensive_risk_score'] >= 55),
        (auto_loans_df['comprehensive_risk_score'] >= 40),
        (auto_loans_df['comprehensive_risk_score'] >= 25)
    ]
    choices = ['Excellent Risk', 'Very Low Risk', 'Low Risk', 'Moderate Risk', 'High Risk']
    auto_loans_df['comprehensive_risk_category'] = np.select(conditions, choices, default='Very High Risk')
    
    # Enhanced loan term eligibility based on comprehensive analysis
    conditions = [
        (auto_loans_df['comprehensive_risk_score'] >= 85),  # Excellent
        (auto_loans_df['comprehensive_risk_score'] >= 70),  # Very Low Risk
        (auto_loans_df['comprehensive_risk_score'] >= 55),  # Low Risk
        (auto_loans_df['comprehensive_risk_score'] >= 40),  # Moderate Risk
        (auto_loans_df['comprehensive_risk_score'] >= 25)   # High Risk
    ]
    choices = [96, 84, 72, 60, 48]  # 8, 7, 6, 5, 4 years
    auto_loans_df['max_loan_term'] = np.select(conditions, choices, default=36)
    
    # Enhanced interest rate calculation
    base_rate = 4.99  # Lower base rate for excellent customers
    risk_premium = (100 - auto_loans_df['comprehensive_risk_score']) * 0.18
    auto_loans_df['interest_rate'] = (base_rate + risk_premium).clip(4.99, 29.99)
    
    # Vehicle type recommendation based on loan amount and profile
    conditions = [
        (auto_loans_df['auto_loan_limit'] >= 80000) & (auto_loans_df['comprehensive_risk_score'] >= 80),
        (auto_loans_df['auto_loan_limit'] >= 50000) & (auto_loans_df['comprehensive_risk_score'] >= 70),
        (auto_loans_df['auto_loan_limit'] >= 30000) & (auto_loans_df['comprehensive_risk_score'] >= 60),
        (auto_loans_df['auto_loan_limit'] >= 20000)
    ]
    choices = ['Luxury Vehicle', 'Premium Vehicle', 'Standard Vehicle', 'Economy Vehicle']
    auto_loans_df['recommended_vehicle_type'] = np.select(conditions, choices, default='Used/Basic Vehicle')
    
    # Down payment recommendation
    auto_loans_df['recommended_down_payment_pct'] = np.select([
        (auto_loans_df['comprehensive_risk_score'] >= 80),
        (auto_loans_df['comprehensive_risk_score'] >= 60),
        (auto_loans_df['comprehensive_risk_score'] >= 40)
    ], [10, 15, 20], default=25)
    
    # Save the comprehensive dataset
    output_path = 'data/banks/auto_loans_bank_full.csv'
    auto_loans_df.to_csv(output_path, index=False)
    print(f"Comprehensive auto loans dataset saved to {output_path}")
    
    # Print comprehensive summary statistics
    print("\nComprehensive Auto Loans Dataset Statistics:")
    print(f"Total records: {len(auto_loans_df)} (100% of original)")
    print(f"Total features: {len(auto_loans_df.columns)}")
    print(f"Eligible customers: {auto_loans_df['loan_eligible'].sum()}")
    print(f"Eligibility rate: {(auto_loans_df['loan_eligible'].sum() / len(auto_loans_df) * 100):.2f}%")
    
    print("\nOriginal Credit Data Features:")
    original_features = [col for col in credit_df.columns]
    for i, feature in enumerate(original_features, 1):
        print(f"{i:2d}. {feature}")
    
    print("\nAdditional Computed Features:")
    new_features = [col for col in auto_loans_df.columns if col not in credit_df.columns]
    for i, feature in enumerate(new_features, 1):
        print(f"{i:2d}. {feature}")
    
    print("\nComprehensive Risk Category Distribution:")
    risk_dist = auto_loans_df['comprehensive_risk_category'].value_counts()
    print(risk_dist)
    print("\nRisk Category Percentages:")
    print((risk_dist / len(auto_loans_df) * 100).round(2))
    
    print("\nRecommended Vehicle Type Distribution:")
    vehicle_dist = auto_loans_df['recommended_vehicle_type'].value_counts()
    print(vehicle_dist)
    
    print("\nAverage Interest Rate by Risk Category:")
    print(auto_loans_df.groupby('comprehensive_risk_category')['interest_rate'].mean().round(2))
    
    print("\nLoan Term Distribution:")
    print(auto_loans_df['max_loan_term'].value_counts().sort_index())
    
    print("\nComprehensive loan statistics by credit score range:")
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
        if mask.sum() > 0:
            subset = auto_loans_df.loc[mask]
            print(f"\n{label}:")
            print(f"  Count: {mask.sum()}")
            print(f"  Eligibility Rate: {(subset['loan_eligible'].sum() / len(subset) * 100):.1f}%")
            print(f"  Average Loan Limit: ${subset['auto_loan_limit'].mean():,.2f}")
            print(f"  Average Risk Score: {subset['comprehensive_risk_score'].mean():.2f}")
            print(f"  Average Interest Rate: {subset['interest_rate'].mean():.2f}%")
            print(f"  Average Loan Term: {subset['max_loan_term'].mean():.1f} months")
    
    return auto_loans_df

def main():
    """Generate comprehensive auto loans dataset"""
    print("Generating comprehensive auto loans bank dataset...")
    print("Using ALL rows and ALL columns from credit scoring dataset")
    auto_loans_df = generate_auto_loans_full_dataset()
    
    print(f"\nDataset generation complete!")
    print(f"Final dataset: {len(auto_loans_df)} rows, {len(auto_loans_df.columns)} columns")

if __name__ == "__main__":
    main() 