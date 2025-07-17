import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def generate_tax_id(n, offset=0):
    """Generate n unique tax IDs in format XXX-XX-XXXX"""
    tax_ids = []
    
    for i in range(n):
        # Create absolutely unique tax ID using sequential approach
        # Convert the sequential number to a unique tax ID format
        seq_num = offset + i
        
        # Use a large base to ensure we don't run out of combinations
        # and convert to tax ID format
        
        # Method: treat the 9-digit space as a single large number space
        # and convert each sequential number to XXX-XX-XXXX format
        
        # Calculate the position in the 9-digit space
        # Total possible combinations: 900 * 90 * 9000 = 729,000,000
        
        # Extract digits directly from the sequential number
        temp_num = seq_num
        
        # Last 4 digits (XXXX part): 1000-9999
        zzzz = 1000 + (temp_num % 9000)
        temp_num = temp_num // 9000
        
        # Middle 2 digits (XX part): 10-99  
        yy = 10 + (temp_num % 90)
        temp_num = temp_num // 90
        
        # First 3 digits (XXX part): 100-999
        xxx = 100 + (temp_num % 900)
        
        tax_id = f"{xxx:03d}-{yy:02d}-{zzzz:04d}"
        tax_ids.append(tax_id)
    
    return tax_ids

def generate_dates(n, start_date='2020-01-01'):
    """Generate random dates after start_date"""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    days = np.random.randint(0, (datetime.now() - start).days, n)
    return [start + timedelta(days=int(day)) for day in days]

def calculate_credit_score(row):
    """Calculate credit score based on features with enhanced distribution for all ranges"""
    weights = {
        'income': 0.12,
        'employment': 0.08,
        'credit_history': 0.12,
        'cards': 0.06,
        'debt': 0.15,
        'payment': 0.15,
        'late_payment': 0.12,
        'savings': 0.04,
        'inquiry': 0.05,
        'age': 0.02,
        'expense': 0.04,
        'utilization': 0.10,
        'transaction': 0.02,
        'digital_banking': 0.02,
        'investment': 0.02
    }
    
    # Calculate individual components with more extreme ranges
    income_score = min(100, (row['annual_income'] / 100000) * 100)
    employment_score = min(100, (row['employment_length'] / 15) * 100)
    history_score = min(100, (row['credit_history_length'] / 12) * 100)
    
    # More punitive credit card scoring
    if row['num_credit_cards'] == 0:
        cards_score = 20  # Very low for no credit cards
    elif 1 <= row['num_credit_cards'] <= 4:
        cards_score = 100
    elif 5 <= row['num_credit_cards'] <= 6:
        cards_score = 80
    else:
        cards_score = max(0, 60 - (row['num_credit_cards'] - 6) * 15)
    
    # More severe debt penalty
    debt_score = max(0, 100 - (row['debt_to_income_ratio'] * 120))
    payment_score = row['payment_history']
    
    # More punitive late payment scoring
    late_payment_score = max(0, 100 - row['late_payments'] * 25)
    
    savings_score = min(100, (row['savings_balance'] / 50000) * 100)
    
    # More severe inquiry penalty
    inquiry_score = max(0, 100 - row['credit_inquiries'] * 20)
    
    age_score = min(100, max(0, (row['age'] - 18) * 2))
    expense_score = max(0, 100 - (row['monthly_expenses'] / (row['annual_income'] / 12)) * 80)
    
    # More severe utilization penalty
    utilization_score = max(0, 100 - row['credit_utilization_ratio'] * 120)
    
    transaction_score = min(100, row['avg_monthly_transactions'] * 1.5)
    digital_score = min(100, row['digital_banking_score'])
    investment_score = min(100, (row['investment_balance'] / 60000) * 100)
    
    # Calculate base score
    total_score = (
        income_score * weights['income'] +
        employment_score * weights['employment'] +
        history_score * weights['credit_history'] +
        cards_score * weights['cards'] +
        debt_score * weights['debt'] +
        payment_score * weights['payment'] +
        late_payment_score * weights['late_payment'] +
        savings_score * weights['savings'] +
        inquiry_score * weights['inquiry'] +
        age_score * weights['age'] +
        expense_score * weights['expense'] +
        utilization_score * weights['utilization'] +
        transaction_score * weights['transaction'] +
        digital_score * weights['digital_banking'] +
        investment_score * weights['investment']
    )
    
    # Enhanced distribution to cover all credit score ranges
    # Use a more aggressive spread to ensure all ranges are represented
    
    # Add random component to ensure distribution across all ranges
    random_factor = np.random.uniform(-10, 10)
    adjusted_score = total_score + random_factor
    
    # Create different credit profiles with more balanced distribution
    if adjusted_score < 30:
        # Poor credit (300-549) - 15% of population
        base_score = 300 + (adjusted_score / 30) * 249
        noise = np.random.normal(0, 20)
    elif adjusted_score < 50:
        # Fair credit (550-649) - 20% of population
        base_score = 550 + ((adjusted_score - 30) / 20) * 99
        noise = np.random.normal(0, 15)
    elif adjusted_score < 75:
        # Good credit (650-749) - 40% of population
        base_score = 650 + ((adjusted_score - 50) / 25) * 99
        noise = np.random.normal(0, 12)
    else:
        # Excellent credit (750-850) - 25% of population
        base_score = 750 + ((adjusted_score - 75) / 25) * 100
        noise = np.random.normal(0, 10)
    
    # Add some randomness to avoid perfect determinism
    credit_score = int(base_score + noise)
    
    # Force minimum representation in each range by using modulo approach
    # This ensures we get samples in all ranges
    customer_id_hash = hash(str(row.get('tax_id', ''))) % 100
    if customer_id_hash < 12:  # 12% poor credit
        if credit_score > 549:
            credit_score = np.random.randint(300, 550)
    elif customer_id_hash < 25:  # 13% fair credit  
        if credit_score < 550 or credit_score > 649:
            credit_score = np.random.randint(550, 650)
    
    # Ensure score stays within FICO bounds
    return min(850, max(300, credit_score))

def generate_dataset(n_records, random_seed=42, offset=0):
    """Generate credit scoring dataset with specified number of records"""
    np.random.seed(random_seed)
    
    # Basic customer information - Enhanced for diverse profiles
    data = {
        'tax_id': generate_tax_id(n_records, offset),
        'age': np.random.gamma(shape=8, scale=4, size=n_records) + 18,  # More age variation
        'employment_length': np.random.gamma(shape=3, scale=5, size=n_records),  # More employment variation
        'annual_income': np.exp(np.random.normal(10.8, 0.9, n_records)),  # More income variation
    }
    
    # Credit and payment history - Enhanced for diverse credit profiles
    data.update({
        'credit_history_length': np.random.gamma(shape=2, scale=6, size=n_records),  # More variation in credit history
        'num_credit_cards': np.random.poisson(2.8, n_records),  # Slightly higher average
        'payment_history': np.random.beta(3, 2, n_records) * 100,  # More variation in payment history
        'late_payments': np.random.poisson(1.2, n_records),  # More late payments for diversity
        'credit_inquiries': np.random.poisson(1.0, n_records),  # More inquiries
        'total_credit_limit': data['annual_income'] * np.random.beta(3, 3, n_records),  # More varied credit limits
        'credit_utilization_ratio': np.random.beta(1.5, 2, n_records),  # Higher utilization rates
        'last_late_payment_days': np.random.exponential(180, n_records),
        'num_loan_accounts': np.random.poisson(1.8, n_records),  # More loan accounts
    })
    
    # Banking behavior
    data.update({
        'savings_balance': np.exp(np.random.normal(9, 1.1, n_records)),
        'checking_balance': np.exp(np.random.normal(8, 0.9, n_records)),
        'investment_balance': np.exp(np.random.normal(8, 1.6, n_records)),
        'avg_monthly_transactions': np.random.gamma(shape=5, scale=11, size=n_records),
        'avg_transaction_value': np.exp(np.random.normal(4, 0.6, n_records)),
        'international_transactions_ratio': np.random.beta(2, 8, n_records),
    })
    
    # Digital banking metrics
    data.update({
        'digital_banking_score': np.random.beta(5, 2, n_records) * 100,
        'mobile_banking_usage': np.random.beta(4, 2, n_records) * 100,
        'online_transactions_ratio': np.random.beta(4, 2, n_records),
        'e_statement_enrolled': np.random.choice([0, 1], size=n_records, p=[0.2, 0.8]),
    })
    
    # Financial obligations
    data.update({
        'current_debt': np.exp(np.random.normal(10, 1.1, n_records)),
        'monthly_expenses': None,
        'auto_loan_balance': np.random.exponential(22000, n_records),
        'mortgage_balance': np.random.exponential(220000, n_records),
        'investment_loan_balance': np.random.exponential(55000, n_records),
    })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Clip values to realistic ranges - Enhanced for diverse profiles
    df['age'] = df['age'].clip(18, 85)
    df['employment_length'] = df['employment_length'].clip(0, 45)  # Allow longer employment
    df['annual_income'] = df['annual_income'].clip(15000, 1000000)  # Wider income range
    df['credit_history_length'] = df['credit_history_length'].clip(0, 40)  # Longer credit history
    df['num_credit_cards'] = df['num_credit_cards'].clip(0, 15)  # More credit cards possible
    df['num_loan_accounts'] = df['num_loan_accounts'].clip(0, 8)  # More loan accounts
    df['total_credit_limit'] = df['total_credit_limit'].clip(1000, 750000)  # Wider credit limit range
    df['credit_utilization_ratio'] = df['credit_utilization_ratio'].clip(0, 1)
    df['last_late_payment_days'] = df['last_late_payment_days'].clip(0, 365)
    df['savings_balance'] = df['savings_balance'].clip(0, 2000000)  # Higher savings possible
    df['checking_balance'] = df['checking_balance'].clip(0, 200000)  # Higher checking balance
    df['investment_balance'] = df['investment_balance'].clip(0, 3000000)  # Higher investments
    df['avg_monthly_transactions'] = df['avg_monthly_transactions'].clip(0, 300)  # More transactions
    df['avg_transaction_value'] = df['avg_transaction_value'].clip(1, 3000)  # Wider transaction range
    df['auto_loan_balance'] = df['auto_loan_balance'].clip(0, 150000)  # Higher auto loans
    df['mortgage_balance'] = df['mortgage_balance'].clip(0, 2000000)  # Higher mortgages
    df['investment_loan_balance'] = df['investment_loan_balance'].clip(0, 500000)  # Higher investment loans
    
    # Calculate derived metrics
    expense_ratios = np.random.beta(6, 4, n_records)
    df['monthly_expenses'] = (df['annual_income'] / 12) * expense_ratios
    
    total_debt = (df['current_debt'] + df['auto_loan_balance'] + 
                 df['mortgage_balance'] + df['investment_loan_balance'])
    # Enhanced debt-to-income ratio for more diverse credit profiles
    base_ratio = np.random.beta(2, 4, n_records) * 0.7  # More variation, some higher ratios
    debt_influence = np.clip(total_debt / df['annual_income'], 0, 1.2) * 0.4  # Allow higher debt influence
    df['debt_to_income_ratio'] = (base_ratio + debt_influence).clip(0, 1.0)  # Allow up to 100% DTI
    
    # Calculate credit scores
    df['credit_score'] = df.apply(calculate_credit_score, axis=1)
    
    # Round monetary values
    monetary_columns = [
        'annual_income', 'current_debt', 'savings_balance', 'checking_balance',
        'investment_balance', 'monthly_expenses', 'total_credit_limit', 
        'avg_transaction_value', 'auto_loan_balance', 'mortgage_balance',
        'investment_loan_balance'
    ]
    df[monetary_columns] = df[monetary_columns].round(2)
    
    # Round percentage and ratio columns
    percentage_columns = [
        'payment_history', 'credit_utilization_ratio', 'digital_banking_score',
        'mobile_banking_usage', 'online_transactions_ratio', 
        'international_transactions_ratio', 'debt_to_income_ratio'
    ]
    df[percentage_columns] = df[percentage_columns].round(4)
    
    # Round other numeric columns
    df['employment_length'] = df['employment_length'].round(1)
    df['credit_history_length'] = df['credit_history_length'].round(1)
    df['avg_monthly_transactions'] = df['avg_monthly_transactions'].round(1)
    df['last_late_payment_days'] = df['last_late_payment_days'].round(0)
    
    return df

def print_dataset_stats(df, dataset_name):
    """Print statistics for a dataset"""
    print(f"\n{dataset_name} Statistics:")
    print(f"Number of records: {len(df)}")
    print("\nCredit score statistics:")
    print(df['credit_score'].describe())
    
    # Print credit score distribution by ranges
    print("\nCredit Score Distribution by Ranges:")
    ranges = [
        (300, 549, "Poor"),
        (550, 649, "Fair"), 
        (650, 749, "Good"),
        (750, 850, "Excellent")
    ]
    
    for min_score, max_score, range_name in ranges:
        count = len(df[(df['credit_score'] >= min_score) & (df['credit_score'] <= max_score)])
        percentage = (count / len(df)) * 100
        print(f"{range_name} ({min_score}-{max_score}): {count:,} records ({percentage:.1f}%)")
    
    if dataset_name == "Training Dataset":
        print("\nSample records from each range:")
        for min_score, max_score, range_name in ranges:
            range_df = df[(df['credit_score'] >= min_score) & (df['credit_score'] <= max_score)]
            if len(range_df) > 0:
                sample = range_df.sample(min(3, len(range_df)))
                print(f"\n{range_name} Range Examples:")
                display_columns = [
                    'tax_id', 'credit_score', 'annual_income', 'credit_utilization_ratio',
                    'debt_to_income_ratio', 'payment_history', 'late_payments'
                ]
                for _, row in sample.iterrows():
                    print(f"  Score: {row['credit_score']}, Income: ${row['annual_income']:,.0f}, "
                          f"Utilization: {row['credit_utilization_ratio']:.2f}, "
                          f"DTI: {row['debt_to_income_ratio']:.2f}, "
                          f"Payment History: {row['payment_history']:.1f}%, "
                          f"Late Payments: {row['late_payments']}")
            else:
                print(f"\n{range_name} Range: No samples generated")

def main():
    """Generate credit scoring datasets"""
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/test', exist_ok=True)
    
    # Generate training dataset
    print("Generating training dataset...")
    train_df = generate_dataset(n_records=500000, random_seed=RANDOM_SEED)
    train_df.to_csv('data/credit_scoring_dataset.csv', index=False)
    print_dataset_stats(train_df, "Training Dataset")
    
    # Generate test dataset
    print("\nGenerating test dataset...")
    test_df = generate_dataset(n_records=50000, random_seed=RANDOM_SEED+1, offset=500000)
    test_df.to_csv('data/test/credit_scoring_test.csv', index=False)
    print_dataset_stats(test_df, "Test Dataset")
    
    print("\nDatasets generated successfully!")

if __name__ == "__main__":
    main() 