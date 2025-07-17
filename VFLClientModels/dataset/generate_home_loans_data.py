import pandas as pd
import numpy as np
import os

def calculate_home_loan_limit(row):
    """
    Calculate home loan limit based on mortgage lending criteria
    Returns loan limit, eligibility status, and loan-to-value ratio
    
    Home Loan Eligibility Rules (Conservative for ~70% approval):
    1. Base Qualification:
       - Credit score must be > 580 (FHA minimum)
       - Payment history must be > 60% (higher than auto loans)
       - Employment length >= 2 years (stable employment)
       - Age >= 21 and <= 65 (working age for long-term loans)
       - Debt-to-income ratio must be < 0.43 (standard DTI limit)
       - Must have sufficient down payment capacity (savings)
       
    2. Loan Amount Calculation:
       - Base on income multiples (conservative approach):
         * 5.0x income if credit score > 760 (Excellent)
         * 4.5x income if credit score > 720 (Very Good)  
         * 4.0x income if credit score > 680 (Good)
         * 3.5x income if credit score > 640 (Fair)
         * 3.0x income if credit score > 600 (Poor)
         * 2.5x income if credit score > 580 (Subprime)
       
    3. Additional Constraints:
       - Maximum DTI after mortgage: 43%
       - Minimum down payment: 3.5% (FHA) to 20% (Conventional)
       - Property value estimated as loan amount / (1 - down payment %)
       - Loan-to-Value ratio calculated
    """
    # Extract features
    annual_income = row['annual_income']
    credit_score = row['credit_score']
    payment_history = row['payment_history']
    employment_length = row['employment_length']
    debt_ratio = row['debt_to_income_ratio']
    age = row['age']
    savings_balance = row['savings_balance']
    checking_balance = row['checking_balance']
    investment_balance = row['investment_balance']
    current_debt = row['current_debt']
    mortgage_balance = row['mortgage_balance']
    
    # Very lenient eligibility criteria for home loans (targeting ~80% approval)
    is_eligible = (
        credit_score > 450 and  # Much lower credit requirement (was 580)
        payment_history > 30.0 and  # Much lower payment history (was 60.0)
        employment_length >= 0.5 and  # Lower employment requirement (was 1.0)
        age >= 18 and age <= 70 and  # Broader age range (was 21-65)
        debt_ratio < 0.95 and  # Very high DTI tolerance (was 0.80)
        mortgage_balance < 500000  # Allow much higher existing mortgages (was 100000)
    )
    
    # Calculate maximum loan amount based on credit score (more generous)
    if credit_score > 760:
        income_multiplier = 6.0  # Increased from 5.0
        min_down_payment_pct = 0.05  # Reduced from 0.10
    elif credit_score > 720:
        income_multiplier = 5.5  # Increased from 4.5
        min_down_payment_pct = 0.05  # Reduced from 0.15
    elif credit_score > 680:
        income_multiplier = 5.0  # Increased from 4.0
        min_down_payment_pct = 0.05  # Reduced from 0.15
    elif credit_score > 640:
        income_multiplier = 4.5  # Increased from 3.5
        min_down_payment_pct = 0.10  # Reduced from 0.20
    elif credit_score > 600:
        income_multiplier = 4.0  # Increased from 3.0
        min_down_payment_pct = 0.10  # Reduced from 0.20
    elif credit_score > 550:
        income_multiplier = 3.5  # New bracket
        min_down_payment_pct = 0.035  # FHA minimum
    elif credit_score > 500:
        income_multiplier = 3.0  # New bracket
        min_down_payment_pct = 0.035  # FHA minimum
    else:  # 450-500 (very subprime)
        income_multiplier = 2.5  # Was for 580-600
        min_down_payment_pct = 0.035  # FHA minimum
    
    # Calculate base loan amount
    max_loan_amount = annual_income * income_multiplier
    
    # Check DTI constraint (new mortgage payment + existing debt) - Very lenient
    # Estimate monthly mortgage payment (principal + interest + taxes + insurance)
    # Using 30-year fixed rate assumption (~6.5% current rates)
    estimated_monthly_payment = max_loan_amount * 0.0045  # Reduced from 0.007 (more realistic)
    total_monthly_debt = (current_debt / 12) + estimated_monthly_payment
    total_dti = total_monthly_debt / (annual_income / 12)
    
    if total_dti > 0.95:  # Very lenient DTI limit (was 0.65)
        # Reduce loan amount to meet DTI requirement
        max_allowed_debt = (annual_income / 12) * 0.95
        available_for_mortgage = max_allowed_debt - (current_debt / 12)
        max_loan_amount = min(max_loan_amount, max(0, available_for_mortgage / 0.0045))  # Ensure positive
    
    # Calculate required down payment and available funds
    available_funds = savings_balance + checking_balance + (investment_balance * 0.8)  # 80% of investments liquid
    estimated_property_value = max_loan_amount / (1 - min_down_payment_pct)
    required_down_payment = estimated_property_value * min_down_payment_pct
    
    # Check if customer has sufficient down payment - be more flexible
    if available_funds < required_down_payment and available_funds > 1000:  # Only apply if they have some funds
        # Reduce property value to match available funds
        affordable_property_value = available_funds / min_down_payment_pct
        max_loan_amount = affordable_property_value * (1 - min_down_payment_pct)
        estimated_property_value = affordable_property_value
    elif available_funds <= 1000:  # Very low funds - use income-based minimum
        # For customers with minimal savings, base on income only
        min_income_based = annual_income * 2.0  # Conservative 2x income minimum
        max_loan_amount = max(max_loan_amount, min_income_based)
        estimated_property_value = max_loan_amount / (1 - min_down_payment_pct)
    
    # More realistic minimum values based on income
    if annual_income >= 100000:
        min_loan = 100000  # Higher earners get higher minimums
    elif annual_income >= 75000:
        min_loan = 75000
    elif annual_income >= 50000:
        min_loan = 50000
    elif annual_income >= 35000:
        min_loan = 35000
    else:
        min_loan = 25000  # Very low income gets lower minimum
        
    max_loan_amount = max(min_loan, max_loan_amount)
    estimated_property_value = max(max_loan_amount / 0.97, estimated_property_value)  # Ensure property value makes sense
    
    # Calculate final loan-to-value ratio
    ltv_ratio = max_loan_amount / estimated_property_value
    
    # Round to nearest thousand
    max_loan_amount = round(max_loan_amount / 1000) * 1000
    estimated_property_value = round(estimated_property_value / 1000) * 1000
    
    return max_loan_amount, is_eligible, ltv_ratio, estimated_property_value, min_down_payment_pct

def generate_home_loans_dataset():
    """Generate home loans dataset from credit scoring data"""
    # Create output directory if it doesn't exist
    os.makedirs('data/banks', exist_ok=True)
    
    # Read credit scoring dataset
    credit_df = pd.read_csv('data/credit_scoring_dataset.csv')
    
    # Randomly select 75% of records for home loans (different seed)
    np.random.seed(456)  # Different seed from auto and digital
    selected_indices = np.random.random(len(credit_df)) <= 0.75
    credit_df = credit_df[selected_indices].copy()
    
    print("\nHome Loan Dataset Generation - Input Data Summary:")
    print(f"Selected {len(credit_df)} records from original dataset")
    
    # Select relevant features for home loans
    home_loan_features = [
        'tax_id',                       # Customer identifier
        'annual_income',                # Primary factor for loan calculation
        'credit_score',                 # Credit worthiness (critical for mortgages)
        'payment_history',              # Payment reliability
        'employment_length',            # Job stability (important for long-term loans)
        'debt_to_income_ratio',         # Existing debt burden
        'age',                          # Age considerations
        'credit_history_length',        # Credit maturity
        'num_credit_cards',             # Credit relationships
        'num_loan_accounts',            # Existing loan burden
        'total_credit_limit',           # Credit capacity
        'credit_utilization_ratio',     # Credit usage
        'late_payments',                # Payment behavior
        'credit_inquiries',             # Recent credit activity
        'current_debt',                 # Current debt amount
        'monthly_expenses',             # Monthly obligations
        'savings_balance',              # Down payment source
        'checking_balance',             # Liquid assets
        'investment_balance',           # Additional assets
        'mortgage_balance',             # Existing mortgage
        'auto_loan_balance',            # Other secured debt
        'last_late_payment_days'        # Recent payment behavior
    ]
    
    # Create home loans dataset
    home_loans_df = credit_df[home_loan_features].copy()
    
    # Calculate home loan limits and related metrics
    results = home_loans_df.apply(calculate_home_loan_limit, axis=1)
    home_loans_df['max_loan_amount'] = [result[0] for result in results]
    home_loans_df['loan_eligible'] = [result[1] for result in results]
    home_loans_df['loan_to_value_ratio'] = [result[2] for result in results]
    home_loans_df['estimated_property_value'] = [result[3] for result in results]
    home_loans_df['min_down_payment_pct'] = [result[4] for result in results]
    
    # Calculate additional home loan specific metrics
    home_loans_df['required_down_payment'] = (
        home_loans_df['estimated_property_value'] * home_loans_df['min_down_payment_pct']
    ).round(0)
    
    # Available funds for down payment
    home_loans_df['available_down_payment_funds'] = (
        home_loans_df['savings_balance'] + 
        home_loans_df['checking_balance'] + 
        (home_loans_df['investment_balance'] * 0.8)
    ).round(0)
    
    # Calculate comprehensive risk score for home loans (0-100)
    home_loans_df['mortgage_risk_score'] = (
        # Credit factors (50% weight)
        (home_loans_df['credit_score'] - 300) / 550 * 25 +  # Credit score (25%)
        (home_loans_df['payment_history']) * 0.15 +          # Payment history (15%)
        ((12 - home_loans_df['late_payments']).clip(0, 12) / 12 * 100) * 0.10 + # Late payments (10%)
        
        # Debt and income factors (25% weight)
        ((1 - home_loans_df['debt_to_income_ratio']) * 100) * 0.10 +  # DTI ratio (10%)
        ((1 - home_loans_df['credit_utilization_ratio']) * 100) * 0.08 + # Credit utilization (8%)
        ((home_loans_df['annual_income'] / 150000).clip(0, 1) * 100) * 0.07 + # Income level (7%)
        
        # Stability factors (15% weight)
        (np.minimum(home_loans_df['employment_length'], 15) / 15 * 100) * 0.08 + # Employment (8%)
        (np.minimum(home_loans_df['credit_history_length'], 25) / 25 * 100) * 0.07 + # Credit history (7%)
        
        # Asset factors (10% weight)
        ((home_loans_df['available_down_payment_funds'] / 100000).clip(0, 1) * 100) * 0.05 + # Down payment funds (5%)
        ((home_loans_df['investment_balance'] / 50000).clip(0, 1) * 100) * 0.03 + # Investment assets (3%)
        ((500000 - home_loans_df['current_debt']).clip(0, 500000) / 500000 * 100) * 0.02 # Debt burden (2%)
    ).clip(0, 100).round(2)
    
    # Risk category based on mortgage risk score
    conditions = [
        (home_loans_df['mortgage_risk_score'] >= 85),
        (home_loans_df['mortgage_risk_score'] >= 70),
        (home_loans_df['mortgage_risk_score'] >= 55),
        (home_loans_df['mortgage_risk_score'] >= 40)
    ]
    choices = ['Excellent Risk', 'Good Risk', 'Acceptable Risk', 'High Risk']
    home_loans_df['risk_category'] = np.select(conditions, choices, default='Very High Risk')
    
    # Calculate interest rates based on risk and current market (2024 rates)
    base_rate = 6.5  # Current 30-year fixed rate baseline
    risk_premium = (100 - home_loans_df['mortgage_risk_score']) * 0.08  # Risk-based pricing
    home_loans_df['interest_rate'] = (base_rate + risk_premium).clip(6.0, 12.0).round(3)
    
    # Loan terms (most home loans are 30-year)
    conditions = [
        (home_loans_df['mortgage_risk_score'] >= 80),  # Excellent risk
        (home_loans_df['mortgage_risk_score'] >= 65),  # Good risk
        (home_loans_df['mortgage_risk_score'] >= 50)   # Acceptable risk
    ]
    choices = [30, 30, 30]  # All 30-year terms
    home_loans_df['loan_term_years'] = np.select(conditions, choices, default=15)  # High risk gets 15-year only
    
    # Loan type recommendation
    conditions = [
        (home_loans_df['credit_score'] >= 720) & (home_loans_df['min_down_payment_pct'] >= 0.20),
        (home_loans_df['credit_score'] >= 680) & (home_loans_df['min_down_payment_pct'] >= 0.15),
        (home_loans_df['credit_score'] >= 620) & (home_loans_df['min_down_payment_pct'] >= 0.10),
        (home_loans_df['credit_score'] >= 580)
    ]
    choices = ['Conventional', 'Conventional', 'VA/USDA Eligible', 'FHA']
    home_loans_df['recommended_loan_type'] = np.select(conditions, choices, default='Hard Money/Alternative')
    
    # Calculate estimated monthly payment (Principal + Interest + Taxes + Insurance)
    monthly_rate = home_loans_df['interest_rate'] / 100 / 12
    num_payments = home_loans_df['loan_term_years'] * 12
    
    # Monthly P&I calculation
    monthly_pi = home_loans_df['max_loan_amount'] * (
        monthly_rate * (1 + monthly_rate) ** num_payments
    ) / ((1 + monthly_rate) ** num_payments - 1)
    
    # Add estimated taxes and insurance (roughly 1.2% of property value annually, reduced from 1.5%)
    monthly_ti = home_loans_df['estimated_property_value'] * 0.012 / 12
    
    home_loans_df['estimated_monthly_payment'] = (monthly_pi + monthly_ti).round(0)
    
    # Calculate debt-to-income after mortgage
    home_loans_df['dti_after_mortgage'] = (
        (home_loans_df['current_debt'] / 12 + home_loans_df['estimated_monthly_payment']) / 
        (home_loans_df['annual_income'] / 12)
    ).round(3)
    
    # Save the dataset
    output_path = 'data/banks/home_loans_bank.csv'
    home_loans_df.to_csv(output_path, index=False)
    print(f"\nHome loans dataset saved to {output_path}")
    
    # Print comprehensive summary statistics
    print("\n" + "="*80)
    print("HOME LOANS DATASET SUMMARY")
    print("="*80)
    print(f"Total records: {len(home_loans_df)}")
    print(f"Eligible customers: {home_loans_df['loan_eligible'].sum()}")
    print(f"Ineligible customers: {(~home_loans_df['loan_eligible']).sum()}")
    print(f"Eligibility rate: {(home_loans_df['loan_eligible'].sum() / len(home_loans_df) * 100):.2f}%")
    
    print(f"\nLoan Amount Statistics (All Customers):")
    print(home_loans_df['max_loan_amount'].describe())
    
    print(f"\nProperty Value Statistics:")
    print(home_loans_df['estimated_property_value'].describe())
    
    print(f"\nRisk Category Distribution:")
    risk_dist = home_loans_df['risk_category'].value_counts()
    print(risk_dist)
    print(f"\nRisk Category Percentages:")
    print((risk_dist / len(home_loans_df) * 100).round(2))
    
    print(f"\nLoan Type Distribution:")
    loan_type_dist = home_loans_df['recommended_loan_type'].value_counts()
    print(loan_type_dist)
    
    print(f"\nAverage Interest Rate by Risk Category:")
    print(home_loans_df.groupby('risk_category')['interest_rate'].mean().round(3))
    
    print(f"\nAverage Loan Amount by Credit Score Range:")
    credit_ranges = [
        (760, 850, 'Excellent (760+)'),
        (720, 760, 'Very Good (720-760)'),
        (680, 720, 'Good (680-720)'),
        (640, 680, 'Fair (640-680)'),
        (600, 640, 'Poor (600-640)'),
        (550, 600, 'Subprime (550-600)'),
        (500, 550, 'Very Poor (500-550)'),
        (450, 500, 'Deep Subprime (450-500)'),
        (300, 450, 'Below Minimum (<450)')
    ]
    
    for min_score, max_score, label in credit_ranges:
        mask = (home_loans_df['credit_score'] > min_score) & (home_loans_df['credit_score'] <= max_score)
        if mask.sum() > 0:
            subset = home_loans_df.loc[mask]
            eligible_count = subset['loan_eligible'].sum()
            
            print(f"\n{label}:")
            print(f"  Count: {mask.sum()}")
            print(f"  Eligible: {eligible_count} ({eligible_count/len(subset)*100:.1f}%)")
            if eligible_count > 0:
                eligible_subset = subset[subset['loan_eligible']]
                print(f"  Avg Loan Amount: ${eligible_subset['max_loan_amount'].mean():,.0f}")
                print(f"  Avg Property Value: ${eligible_subset['estimated_property_value'].mean():,.0f}")
                print(f"  Avg Interest Rate: {eligible_subset['interest_rate'].mean():.2f}%")
                print(f"  Avg Monthly Payment: ${eligible_subset['estimated_monthly_payment'].mean():,.0f}")
                print(f"  Avg LTV: {eligible_subset['loan_to_value_ratio'].mean():.1%}")
    
    print(f"\nDown Payment Analysis:")
    print(f"Average Required Down Payment: ${home_loans_df['required_down_payment'].mean():,.0f}")
    print(f"Average Available Funds: ${home_loans_df['available_down_payment_funds'].mean():,.0f}")
    
    sufficient_funds = home_loans_df['available_down_payment_funds'] >= home_loans_df['required_down_payment']
    print(f"Customers with Sufficient Down Payment: {sufficient_funds.sum()} ({sufficient_funds.mean()*100:.1f}%)")
    
    print(f"\nDebt-to-Income Analysis:")
    print(f"Average DTI Before Mortgage: {home_loans_df['debt_to_income_ratio'].mean():.1%}")
    print(f"Average DTI After Mortgage: {home_loans_df['dti_after_mortgage'].mean():.1%}")
    
    acceptable_dti = home_loans_df['dti_after_mortgage'] <= 0.95  # Updated to match new very lenient limit
    print(f"Customers with Acceptable DTI After Mortgage: {acceptable_dti.sum()} ({acceptable_dti.mean()*100:.1f}%)")
    
    return home_loans_df

def print_detailed_loan_examples(df, num_examples=5):
    """Print detailed examples of loan calculations"""
    print(f"\n" + "="*80)
    print(f"DETAILED LOAN EXAMPLES ({num_examples} customers)")
    print("="*80)
    
    # Select diverse examples
    sample_df = df.sample(n=num_examples, random_state=789)
    
    for idx, (_, row) in enumerate(sample_df.iterrows(), 1):
        print(f"\nExample {idx} - Customer {row['tax_id']}:")
        print(f"  Credit Score: {row['credit_score']}")
        print(f"  Annual Income: ${row['annual_income']:,.0f}")
        print(f"  Employment Length: {row['employment_length']:.1f} years")
        print(f"  Current DTI: {row['debt_to_income_ratio']:.1%}")
        print(f"  Available Funds: ${row['available_down_payment_funds']:,.0f}")
        print(f"  ")
        print(f"  Loan Eligible: {'Yes' if row['loan_eligible'] else 'No'}")
        if row['loan_eligible']:
            print(f"  Max Loan Amount: ${row['max_loan_amount']:,.0f}")
            print(f"  Property Value: ${row['estimated_property_value']:,.0f}")
            print(f"  Down Payment: ${row['required_down_payment']:,.0f} ({row['min_down_payment_pct']:.1%})")
            print(f"  LTV Ratio: {row['loan_to_value_ratio']:.1%}")
            print(f"  Interest Rate: {row['interest_rate']:.3f}%")
            print(f"  Monthly Payment: ${row['estimated_monthly_payment']:,.0f}")
            print(f"  DTI After Mortgage: {row['dti_after_mortgage']:.1%}")
            print(f"  Risk Category: {row['risk_category']}")
            print(f"  Loan Type: {row['recommended_loan_type']}")

if __name__ == "__main__":
    print("🏠 Home Loans Dataset Generator")
    print("="*50)
    
    # Generate the dataset
    home_loans_df = generate_home_loans_dataset()
    
    # Print detailed examples
    print_detailed_loan_examples(home_loans_df)
    
    print(f"\n✅ Home loans dataset generation completed!")
    print(f"📁 Dataset saved to: data/banks/home_loans_bank.csv")
    print(f"📊 Total records: {len(home_loans_df)}")
    print(f"🎯 Eligible customers: {home_loans_df['loan_eligible'].sum()}") 