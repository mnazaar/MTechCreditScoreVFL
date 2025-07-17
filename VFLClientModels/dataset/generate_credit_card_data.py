import pandas as pd
import numpy as np
import os

def calculate_credit_card_limit(row):
    """
    Calculate credit card limit based on comprehensive criteria
    Returns credit limit, eligibility status, and card tier
    
    Eligibility Rules (LIBERAL for ~85-90% approval):
    1. Basic Qualification:
       - Credit score must be > 400 (very liberal for credit cards)
       - Payment history must be > 0.25 (25% - very liberal)
       - Employment length >= 0.1 year (1+ months)
       - Age >= 18
       - Debt-to-income ratio must be < 0.95 (very high tolerance)
       - No bankruptcy in recent history (simulated)
       
    2. Card Tier and Multiplier based on credit profile:
       - Platinum (Premium): Credit score > 750, Income > $75k, DTI < 0.30
       - Gold (Rewards): Credit score > 680, Income > $45k, DTI < 0.50
       - Silver (Standard): Credit score > 600, Income > $25k, DTI < 0.70
       - Basic (Entry): Credit score > 500, Income > $15k, DTI < 0.85
       - Secured (Subprime): Credit score > 400, Any income, DTI < 0.95
       
    3. Limit Calculation:
       - Platinum: 3.0-8.0x monthly income
       - Gold: 2.0-5.0x monthly income  
       - Silver: 1.5-3.0x monthly income
       - Basic: 1.0-2.0x monthly income
       - Secured: 0.5-1.0x monthly income (often requires deposit)
       
    4. Adjustments:
       - Reduce by 30% if credit_utilization_ratio > 0.80
       - Reduce by 20% if DTI > 0.60
       - Reduce by 15% for each recent credit inquiry > 3
       - Reduce by 10% for each late payment > 2
       - Add 20% if payment_history > 95%
       - Add 15% if employment_length > 5 years
       - Add 10% if existing relationship (has other loans)
    """
    # Extract features
    annual_income = row['annual_income']
    monthly_income = annual_income / 12
    credit_score = row['credit_score']
    payment_history = row['payment_history']
    employment_length = row['employment_length']
    debt_ratio = row['debt_to_income_ratio']
    util_ratio = row['credit_utilization_ratio']
    late_payments = row['late_payments']
    credit_inquiries = row['credit_inquiries']
    age = row['age']
    credit_history_length = row['credit_history_length']
    num_credit_cards = row['num_credit_cards']
    total_credit_limit = row['total_credit_limit']
    
    # Liberal eligibility criteria for high approval rates
    is_eligible = (
        credit_score > 400 and  # Very low credit score requirement
        payment_history > 25.0 and  # Very low payment history requirement (25%)
        employment_length >= 0.1 and  # Very low employment requirement (1+ months)
        age >= 18 and  # Standard age requirement
        debt_ratio < 0.95  # Very high DTI tolerance
    )
    
    # Determine card tier and base multiplier
    if (credit_score > 750 and annual_income > 75000 and debt_ratio < 0.30):
        card_tier = "Platinum"
        min_multiplier = 3.0
        max_multiplier = 8.0
        base_limit = 10000  # High base limit
    elif (credit_score > 680 and annual_income > 45000 and debt_ratio < 0.50):
        card_tier = "Gold" 
        min_multiplier = 2.0
        max_multiplier = 5.0
        base_limit = 5000
    elif (credit_score > 600 and annual_income > 25000 and debt_ratio < 0.70):
        card_tier = "Silver"
        min_multiplier = 1.5
        max_multiplier = 3.0
        base_limit = 2000
    elif (credit_score > 500 and annual_income > 15000 and debt_ratio < 0.85):
        card_tier = "Basic"
        min_multiplier = 1.0
        max_multiplier = 2.0
        base_limit = 1000
    else:  # Secured card for subprime
        card_tier = "Secured"
        min_multiplier = 0.5
        max_multiplier = 1.0
        base_limit = 500
    
    # Calculate multiplier based on credit profile within tier range
    score_factor = (credit_score - 300) / 550  # Normalize credit score
    income_factor = min(1.0, annual_income / 100000)  # Income factor
    history_factor = min(1.0, credit_history_length / 20)  # Credit history factor
    
    # Weighted average of factors
    profile_strength = (score_factor * 0.5 + income_factor * 0.3 + history_factor * 0.2)
    multiplier = min_multiplier + (max_multiplier - min_multiplier) * profile_strength
    
    # Calculate initial credit limit
    credit_limit = max(base_limit, monthly_income * multiplier)
    
    # Apply adjustments for all customers (positive and negative)
    adjustment_factor = 1.0
    
    # Negative adjustments
    if util_ratio > 0.80:
        adjustment_factor *= 0.70  # Reduce by 30%
    if debt_ratio > 0.60:
        adjustment_factor *= 0.80  # Reduce by 20%
    if credit_inquiries > 3:
        excess_inquiries = credit_inquiries - 3
        adjustment_factor *= (0.85 ** excess_inquiries)  # Reduce by 15% per excess inquiry
    if late_payments > 2:
        excess_late = late_payments - 2
        adjustment_factor *= (0.90 ** excess_late)  # Reduce by 10% per excess late payment
    
    # Positive adjustments
    if payment_history > 95.0:
        adjustment_factor *= 1.20  # Add 20%
    if employment_length > 5:
        adjustment_factor *= 1.15  # Add 15%
    if (row['auto_loan_balance'] > 0 or row['mortgage_balance'] > 0):
        adjustment_factor *= 1.10  # Add 10% for existing relationship
    
    # Apply adjustments
    credit_limit *= adjustment_factor
    
    # Card tier specific minimums and maximums
    tier_limits = {
        "Platinum": (10000, 100000),
        "Gold": (5000, 50000),
        "Silver": (1000, 25000),
        "Basic": (500, 10000),
        "Secured": (200, 2000)
    }
    
    min_limit, max_limit = tier_limits[card_tier]
    credit_limit = max(min_limit, min(max_limit, credit_limit))
    
    # Round to nearest $100
    credit_limit = round(credit_limit / 100) * 100
    
    return credit_limit, is_eligible, card_tier

def calculate_interest_rate(row, card_tier):
    """Calculate interest rate based on credit profile and card tier"""
    credit_score = row['credit_score']
    
    # Base rates by tier
    base_rates = {
        "Platinum": 12.99,
        "Gold": 15.99,
        "Silver": 18.99,
        "Basic": 22.99,
        "Secured": 24.99
    }
    
    base_rate = base_rates[card_tier]
    
    # Adjust based on credit score within tier
    if credit_score >= 800:
        rate_adjustment = -3.0  # Excellent credit gets discount
    elif credit_score >= 750:
        rate_adjustment = -1.5
    elif credit_score >= 700:
        rate_adjustment = 0.0
    elif credit_score >= 650:
        rate_adjustment = 1.5
    elif credit_score >= 600:
        rate_adjustment = 3.0
    elif credit_score >= 550:
        rate_adjustment = 4.5
    else:
        rate_adjustment = 6.0  # Poor credit gets penalty
    
    # Additional adjustments
    if row['debt_to_income_ratio'] > 0.70:
        rate_adjustment += 2.0
    if row['late_payments'] > 5:
        rate_adjustment += 1.5
    if row['payment_history'] > 95:
        rate_adjustment -= 1.0
    
    final_rate = base_rate + rate_adjustment
    return round(max(9.99, min(29.99, final_rate)), 2)

def calculate_annual_fee(card_tier, credit_limit):
    """Calculate annual fee based on card tier and limit"""
    if card_tier == "Platinum":
        if credit_limit >= 50000:
            return 495  # Premium platinum
        else:
            return 295  # Standard platinum
    elif card_tier == "Gold":
        if credit_limit >= 20000:
            return 95   # Premium gold
        else:
            return 0    # No-fee gold
    else:
        return 0  # Silver, Basic, and Secured have no annual fee

def generate_credit_card_dataset():
    """Generate credit card dataset from credit scoring data"""
    # Create output directory if it doesn't exist
    os.makedirs('data/banks', exist_ok=True)
    
    # Read credit scoring dataset
    credit_df = pd.read_csv('data/credit_scoring_dataset.csv')
    
    # Randomly select 75% of records with different seed for credit cards
    np.random.seed(456)  # Different seed from other banks
    selected_indices = np.random.random(len(credit_df)) <= 0.75
    credit_df = credit_df[selected_indices].copy()
    
    print("\nCredit Score Distribution for Credit Cards:")
    print(credit_df['credit_score'].describe())
    
    print("\nPayment History Distribution:")
    print(credit_df['payment_history'].describe())
    
    print("\nDebt-to-Income Ratio Distribution:")
    print(credit_df['debt_to_income_ratio'].describe())
    
    print("\nCredit Utilization Distribution:")
    print(credit_df['credit_utilization_ratio'].describe())
    
    # Select relevant features for credit cards
    credit_card_features = [
        'tax_id',                       # Customer identifier
        'annual_income',                # Primary factor for limit calculation
        'credit_score',                 # Most important for credit cards
        'payment_history',              # Payment reliability
        'employment_length',            # Income stability
        'debt_to_income_ratio',         # Existing debt burden
        'age',                          # Customer age
        'credit_history_length',        # Credit experience
        'num_credit_cards',             # Existing credit card relationships
        'num_loan_accounts',            # Other credit relationships
        'total_credit_limit',           # Current credit exposure
        'credit_utilization_ratio',     # Credit usage behavior
        'late_payments',                # Payment behavior
        'credit_inquiries',             # Recent credit seeking behavior
        'current_debt',                 # Current debt level
        'monthly_expenses',             # Monthly obligations
        'savings_balance',              # Financial cushion
        'checking_balance',             # Liquid assets
        'investment_balance',           # Investment assets
        'auto_loan_balance',            # Existing auto debt
        'mortgage_balance',             # Existing mortgage debt
        'last_late_payment_days'        # Recent payment behavior
    ]
    
    # Create credit card dataset
    credit_card_df = credit_df[credit_card_features].copy()
    
    # Calculate credit card limits, eligibility, and card tiers
    results = credit_card_df.apply(calculate_credit_card_limit, axis=1)
    credit_card_df['credit_card_limit'] = [result[0] for result in results]
    credit_card_df['card_eligible'] = [result[1] for result in results]
    credit_card_df['card_tier'] = [result[2] for result in results]
    
    # Calculate interest rates
    credit_card_df['apr'] = credit_card_df.apply(
        lambda row: calculate_interest_rate(row, row['card_tier']), axis=1
    )
    
    # Calculate annual fees
    credit_card_df['annual_fee'] = credit_card_df.apply(
        lambda row: calculate_annual_fee(row['card_tier'], row['credit_card_limit']), axis=1
    )
    
    # Calculate credit utilization impact score
    credit_card_df['utilization_impact'] = np.where(
        credit_card_df['credit_utilization_ratio'] > 0.30,
        'High Impact',
        np.where(credit_card_df['credit_utilization_ratio'] > 0.10, 'Medium Impact', 'Low Impact')
    )
    
    # Calculate available credit (new limit + existing available credit)
    current_available = credit_card_df['total_credit_limit'] * (1 - credit_card_df['credit_utilization_ratio'])
    credit_card_df['total_available_credit'] = current_available + credit_card_df['credit_card_limit']
    
    # Calculate credit-to-income ratio after new card
    credit_card_df['credit_to_income_ratio'] = (
        (credit_card_df['total_credit_limit'] + credit_card_df['credit_card_limit']) / 
        credit_card_df['annual_income']
    ).round(3)
    
    # Calculate risk score (0-100) - Higher score = lower risk
    credit_card_df['risk_score'] = (
        (credit_card_df['credit_score'] - 300) / 550 * 30 +  # Credit score (30%)
        credit_card_df['payment_history'] * 0.20 +           # Payment history (20%)
        ((1 - credit_card_df['debt_to_income_ratio']) * 100) * 0.15 +  # Debt ratio (15%)
        ((1 - credit_card_df['credit_utilization_ratio']) * 100) * 0.10 + # Utilization (10%)
        (np.minimum(credit_card_df['employment_length'], 10) * 10) * 0.08 + # Employment (8%)
        (np.minimum(credit_card_df['credit_history_length'], 20) / 20 * 100) * 0.07 + # History length (7%)
        ((10 - credit_card_df['late_payments']).clip(0, 10) / 10 * 100) * 0.05 + # Late payments (5%)
        ((5 - credit_card_df['credit_inquiries']).clip(0, 5) / 5 * 100) * 0.03 + # Inquiries (3%)
        ((credit_card_df['savings_balance'] + credit_card_df['checking_balance']) / 25000).clip(0, 1) * 100 * 0.02  # Liquid assets (2%)
    ).clip(0, 100).round(1)
    
    # Add risk category
    conditions = [
        (credit_card_df['risk_score'] >= 80),
        (credit_card_df['risk_score'] >= 65),
        (credit_card_df['risk_score'] >= 50),
        (credit_card_df['risk_score'] >= 35)
    ]
    choices = ['Very Low Risk', 'Low Risk', 'Moderate Risk', 'High Risk']
    credit_card_df['risk_category'] = np.select(conditions, choices, default='Very High Risk')
    
    # Calculate rewards eligibility
    credit_card_df['rewards_eligible'] = np.where(
        (credit_card_df['card_tier'].isin(['Platinum', 'Gold'])) & 
        (credit_card_df['risk_score'] >= 60),
        'Yes', 'No'
    )
    
    # Calculate cash advance limit (typically 20-30% of credit limit)
    credit_card_df['cash_advance_limit'] = (
        credit_card_df['credit_card_limit'] * 
        np.where(credit_card_df['card_tier'] == 'Platinum', 0.30,
        np.where(credit_card_df['card_tier'] == 'Gold', 0.25, 0.20))
    ).round(0)
    
    # Save the dataset
    output_path = 'data/banks/credit_card_bank.csv'
    credit_card_df.to_csv(output_path, index=False)
    print(f"\nCredit card dataset saved to {output_path}")
    
    # Print comprehensive summary statistics
    print("\n" + "="*80)
    print("CREDIT CARD DATASET SUMMARY")
    print("="*80)
    
    print(f"\nDataset Overview:")
    print(f"Total records: {len(credit_card_df):,}")
    print(f"Eligible customers: {credit_card_df['card_eligible'].sum():,}")
    print(f"Ineligible customers: {(~credit_card_df['card_eligible']).sum():,}")
    print(f"Approval rate: {(credit_card_df['card_eligible'].sum() / len(credit_card_df) * 100):.2f}%")
    
    print(f"\nCard Tier Distribution:")
    tier_dist = credit_card_df['card_tier'].value_counts()
    for tier, count in tier_dist.items():
        percentage = (count / len(credit_card_df)) * 100
        print(f"  {tier}: {count:,} customers ({percentage:.2f}%)")
    
    print(f"\nCredit Limit Statistics:")
    print("All Customers:")
    print(f"  Mean: ${credit_card_df['credit_card_limit'].mean():,.2f}")
    print(f"  Median: ${credit_card_df['credit_card_limit'].median():,.2f}")
    print(f"  Min: ${credit_card_df['credit_card_limit'].min():,.2f}")
    print(f"  Max: ${credit_card_df['credit_card_limit'].max():,.2f}")
    
    print("\nEligible Customers Only:")
    eligible_df = credit_card_df[credit_card_df['card_eligible']]
    print(f"  Mean: ${eligible_df['credit_card_limit'].mean():,.2f}")
    print(f"  Median: ${eligible_df['credit_card_limit'].median():,.2f}")
    print(f"  Min: ${eligible_df['credit_card_limit'].min():,.2f}")
    print(f"  Max: ${eligible_df['credit_card_limit'].max():,.2f}")
    
    print(f"\nInterest Rate Statistics:")
    print(f"  Mean APR: {credit_card_df['apr'].mean():.2f}%")
    print(f"  Median APR: {credit_card_df['apr'].median():.2f}%")
    print(f"  Range: {credit_card_df['apr'].min():.2f}% - {credit_card_df['apr'].max():.2f}%")
    
    print(f"\nAnnual Fee Statistics:")
    print(f"  Cards with fees: {(credit_card_df['annual_fee'] > 0).sum():,}")
    print(f"  Cards without fees: {(credit_card_df['annual_fee'] == 0).sum():,}")
    print(f"  Average fee (fee cards only): ${credit_card_df[credit_card_df['annual_fee'] > 0]['annual_fee'].mean():.2f}")
    
    print(f"\nRisk Category Distribution:")
    risk_dist = credit_card_df['risk_category'].value_counts()
    for risk, count in risk_dist.items():
        percentage = (count / len(credit_card_df)) * 100
        print(f"  {risk}: {count:,} ({percentage:.2f}%)")
    
    print(f"\nRewards Eligibility:")
    rewards_dist = credit_card_df['rewards_eligible'].value_counts()
    for status, count in rewards_dist.items():
        percentage = (count / len(credit_card_df)) * 100
        print(f"  {status}: {count:,} ({percentage:.2f}%)")
    
    # Detailed analysis by card tier
    print(f"\n" + "="*80)
    print("DETAILED ANALYSIS BY CARD TIER")
    print("="*80)
    
    for tier in ['Platinum', 'Gold', 'Silver', 'Basic', 'Secured']:
        tier_data = credit_card_df[credit_card_df['card_tier'] == tier]
        if len(tier_data) > 0:
            eligible_count = tier_data['card_eligible'].sum()
            print(f"\n{tier} Card Statistics:")
            print(f"  Total customers: {len(tier_data):,}")
            print(f"  Eligible: {eligible_count:,} ({eligible_count/len(tier_data)*100:.1f}%)")
            print(f"  Average credit limit: ${tier_data['credit_card_limit'].mean():,.2f}")
            print(f"  Average APR: {tier_data['apr'].mean():.2f}%")
            print(f"  Average annual fee: ${tier_data['annual_fee'].mean():.2f}")
            print(f"  Average credit score: {tier_data['credit_score'].mean():.0f}")
            print(f"  Average annual income: ${tier_data['annual_income'].mean():,.2f}")
            print(f"  Average risk score: {tier_data['risk_score'].mean():.1f}")
    
    # Credit score range analysis
    print(f"\n" + "="*80)
    print("ANALYSIS BY CREDIT SCORE RANGE")
    print("="*80)
    
    credit_ranges = [
        (750, 850, 'Excellent (750+)'),
        (700, 750, 'Very Good (700-750)'),
        (650, 700, 'Good (650-700)'),
        (600, 650, 'Fair (600-650)'),
        (550, 600, 'Poor (550-600)'),
        (500, 550, 'Very Poor (500-550)'),
        (300, 500, 'Subprime (<500)')
    ]
    
    for min_score, max_score, label in credit_ranges:
        mask = (credit_card_df['credit_score'] > min_score) & (credit_card_df['credit_score'] <= max_score)
        if mask.sum() > 0:
            subset = credit_card_df.loc[mask]
            eligible_count = subset['card_eligible'].sum()
            
            print(f"\n{label}:")
            print(f"  Total customers: {mask.sum():,}")
            print(f"  Approval rate: {eligible_count/len(subset)*100:.1f}%")
            print(f"  Average credit limit: ${subset['credit_card_limit'].mean():,.2f}")
            print(f"  Average APR: {subset['apr'].mean():.2f}%")
            print(f"  Most common tier: {subset['card_tier'].mode().iloc[0] if len(subset) > 0 else 'N/A'}")
            print(f"  Average risk score: {subset['risk_score'].mean():.1f}")
    
    return credit_card_df

def print_sample_records(df, n_samples=5):
    """Print sample records from each card tier"""
    print(f"\n" + "="*80)
    print("SAMPLE RECORDS BY CARD TIER")
    print("="*80)
    
    for tier in ['Platinum', 'Gold', 'Silver', 'Basic', 'Secured']:
        tier_data = df[df['card_tier'] == tier]
        if len(tier_data) > 0:
            print(f"\n{tier} Card Examples:")
            print("-" * 60)
            sample = tier_data.sample(min(n_samples, len(tier_data)))
            
            for _, row in sample.iterrows():
                print(f"  Customer {row['tax_id']}: "
                      f"Score={row['credit_score']:.0f}, "
                      f"Income=${row['annual_income']:,.0f}, "
                      f"Limit=${row['credit_card_limit']:,.0f}, "
                      f"APR={row['apr']:.1f}%, "
                      f"Fee=${row['annual_fee']:.0f}")

if __name__ == "__main__":
    print("Generating Credit Card Dataset...")
    print("="*50)
    
    # Generate the dataset
    credit_card_df = generate_credit_card_dataset()
    
    # Print sample records
    print_sample_records(credit_card_df)
    
    print(f"\n" + "="*80)
    print("DATASET GENERATION COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"✅ Dataset saved to: data/banks/credit_card_bank.csv")
    print(f"📊 Total records: {len(credit_card_df):,}")
    print(f"🎯 Approval rate: {(credit_card_df['card_eligible'].sum() / len(credit_card_df) * 100):.1f}%")
    print(f"💳 Card tiers: {len(credit_card_df['card_tier'].unique())} different tiers")
    print(f"💰 Credit limit range: ${credit_card_df['credit_card_limit'].min():,.0f} - ${credit_card_df['credit_card_limit'].max():,.0f}") 