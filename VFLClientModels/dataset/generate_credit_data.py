import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def generate_tax_id(n, offset=0):
    """Generate n unique tax IDs in format XXX-XX-XXXX"""
    base_numbers = np.arange(offset, offset + n)
    tax_ids = [f"{(num % 900 + 100):03d}-{((num // 900) % 90 + 10):02d}-{(num % 9000 + 1000):04d}" 
               for num in base_numbers]
    return tax_ids

def generate_dates(n, start_date='2020-01-01'):
    """Generate random dates after start_date"""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    days = np.random.randint(0, (datetime.now() - start).days, n)
    return [start + timedelta(days=int(day)) for day in days]

def calculate_credit_score(row):
    """Calculate credit score based on features with deterministic weights"""
    weights = {
        'income': 0.11,
        'employment': 0.08,
        'credit_history': 0.11,
        'cards': 0.05,
        'debt': 0.13,
        'payment': 0.13,
        'late_payment': 0.11,
        'savings': 0.05,
        'inquiry': 0.04,
        'age': 0.02,
        'expense': 0.03,
        'utilization': 0.08,
        'transaction': 0.02,
        'digital_banking': 0.02,
        'investment': 0.02
    }
    
    # Calculate individual components
    income_score = min(100, (row['annual_income'] / 120000) * 100)
    employment_score = min(100, (row['employment_length'] / 12) * 100)
    history_score = min(100, (row['credit_history_length'] / 10) * 100)
    cards_score = 100 if 1 <= row['num_credit_cards'] <= 5 else max(0, 100 - abs(row['num_credit_cards'] - 3) * 12)
    debt_score = max(0, 100 - (row['debt_to_income_ratio'] * 100))
    payment_score = row['payment_history']
    late_payment_score = max(0, 100 - row['late_payments'] * 18)
    savings_score = min(100, (row['savings_balance'] / 40000) * 100)
    inquiry_score = max(0, 100 - row['credit_inquiries'] * 12)
    age_score = min(100, max(0, (row['age'] - 18) * 2.5))
    expense_score = max(0, 100 - (row['monthly_expenses'] / (row['annual_income'] / 12)) * 70)
    utilization_score = max(0, 100 - row['credit_utilization_ratio'] * 100)
    transaction_score = min(100, row['avg_monthly_transactions'] * 2)
    digital_score = min(100, row['digital_banking_score'])
    investment_score = min(100, (row['investment_balance'] / 40000) * 100)
    
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
    
    # Convert total score to a normal distribution centered at 650
    # Using the properties of normal distribution:
    # - Mean = 650
    # - SD ≈ 75 (to get ~5% above 800 and ~5% below 400)
    mean_score = 650
    sd_score = 75
    
    # Convert total_score (0-100) to z-score
    z_score = (total_score - 50) / 15  # Normalize around 0
    
    # Calculate FICO score using normal distribution
    credit_score = int(mean_score + (z_score * sd_score))
    
    # Ensure score stays within FICO bounds
    return min(850, max(300, credit_score))

def generate_dataset(n_records, random_seed=42, offset=0):
    """Generate credit scoring dataset with specified number of records"""
    np.random.seed(random_seed)
    
    # Basic customer information
    data = {
        'tax_id': generate_tax_id(n_records, offset),
        'age': np.random.gamma(shape=11, scale=3, size=n_records) + 18,
        'employment_length': np.random.gamma(shape=4, scale=4, size=n_records),
        'annual_income': np.exp(np.random.normal(11, 0.7, n_records)),
    }
    
    # Credit and payment history
    data.update({
        'credit_history_length': np.random.gamma(shape=3, scale=4, size=n_records),
        'num_credit_cards': np.random.poisson(2.2, n_records),
        'payment_history': np.random.beta(6, 2, n_records) * 100,
        'late_payments': np.random.poisson(0.7, n_records),
        'credit_inquiries': np.random.poisson(0.7, n_records),
        'total_credit_limit': data['annual_income'] * np.random.beta(5, 2, n_records),  # 0.5x to 2.5x annual income
        'credit_utilization_ratio': np.random.beta(2, 4, n_records),
        'last_late_payment_days': np.random.exponential(150, n_records),
        'num_loan_accounts': np.random.poisson(1.5, n_records),
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
    
    # Clip values to realistic ranges
    df['age'] = df['age'].clip(18, 85)
    df['employment_length'] = df['employment_length'].clip(0, 40)
    df['annual_income'] = df['annual_income'].clip(18000, 800000)
    df['credit_history_length'] = df['credit_history_length'].clip(0, 35)
    df['num_credit_cards'] = df['num_credit_cards'].clip(0, 10)
    df['num_loan_accounts'] = df['num_loan_accounts'].clip(0, 5)
    df['total_credit_limit'] = df['total_credit_limit'].clip(5000, 500000)  # Increased max limit
    df['credit_utilization_ratio'] = df['credit_utilization_ratio'].clip(0, 1)
    df['last_late_payment_days'] = df['last_late_payment_days'].clip(0, 365)
    df['savings_balance'] = df['savings_balance'].clip(0, 1500000)
    df['checking_balance'] = df['checking_balance'].clip(0, 150000)
    df['investment_balance'] = df['investment_balance'].clip(0, 2000000)
    df['avg_monthly_transactions'] = df['avg_monthly_transactions'].clip(0, 250)
    df['avg_transaction_value'] = df['avg_transaction_value'].clip(5, 2000)
    df['auto_loan_balance'] = df['auto_loan_balance'].clip(0, 120000)
    df['mortgage_balance'] = df['mortgage_balance'].clip(0, 1500000)
    df['investment_loan_balance'] = df['investment_loan_balance'].clip(0, 300000)
    
    # Calculate derived metrics
    expense_ratios = np.random.beta(6, 4, n_records)
    df['monthly_expenses'] = (df['annual_income'] / 12) * expense_ratios
    
    total_debt = (df['current_debt'] + df['auto_loan_balance'] + 
                 df['mortgage_balance'] + df['investment_loan_balance'])
    # Adjust debt-to-income ratio to be more realistic (most should be under 0.43)
    base_ratio = np.random.beta(4, 8, n_records) * 0.5  # Most under 0.3
    debt_influence = np.clip(total_debt / df['annual_income'], 0, 1) * 0.3  # Max 0.3 from actual debt
    df['debt_to_income_ratio'] = (base_ratio + debt_influence).clip(0, 0.8)
    
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
    if dataset_name == "Training Dataset":
        print("\nSample records (key features):")
        display_columns = [
            'tax_id', 'credit_score', 'annual_income', 'credit_utilization',
            'savings_balance', 'digital_banking_score', 'debt_to_income'
        ]
        print(df[display_columns].head())

def main():
    """Generate credit scoring datasets"""
    # Create directories
    os.makedirs('VFLClientModels/dataset/data', exist_ok=True)
    os.makedirs('VFLClientModels/dataset/data/test', exist_ok=True)
    
    # Generate training dataset
    print("Generating training dataset...")
    train_df = generate_dataset(n_records=10000, random_seed=RANDOM_SEED)
    train_df.to_csv('VFLClientModels/dataset/data/credit_scoring_dataset.csv', index=False)
    print_dataset_stats(train_df, "Training")
    
    # Generate test dataset
    print("\nGenerating test dataset...")
    test_df = generate_dataset(n_records=2000, random_seed=RANDOM_SEED+1, offset=10000)
    test_df.to_csv('VFLClientModels/dataset/data/test/credit_scoring_test.csv', index=False)
    print_dataset_stats(test_df, "Test")
    
    print("\nDatasets generated successfully!")

if __name__ == "__main__":
    main() 