import pandas as pd
import numpy as np
import os

def determine_savings_category(row):
    """
    Determine customer category based on weighted factors (Enhanced with new features):
    - savings_balance (20%): Higher balances indicate better category
    - investment_balance (15%): Investment portfolio size
    - digital_engagement (15%): Level of digital channel usage
    - credit_score (10%): Overall creditworthiness
    - payment_history (10%): Better payment history indicates reliability
    - annual_income (10%): Higher income indicates better potential
    - transaction_value (8%): Higher transaction values indicate more business
    - employment_stability (5%): Job stability
    - international_activity (3%): Global banking activity
    - age (2%): Longer relationship potential
    - e_statement (1%): Digital adoption
    - mobile_usage (1%): Digital engagement
    
    Categories:
    - Regular: Basic customers
    - Preferred: Mid-tier customers with good financial standing
    - VIP: Top-tier customers with excellent financial metrics
    """
    # Calculate individual component scores (0-100)
    savings_score = min(100, (row['savings_balance'] / 100000) * 100)  # Lowered threshold
    investment_score = min(100, (row['investment_balance'] / 50000) * 100)  # Investment portfolio
    digital_score = row['digital_banking_score']
    credit_score_norm = (row['credit_score'] - 300) / 550 * 100  # Normalize credit score
    payment_score = row['payment_history']
    income_score = min(100, (row['annual_income'] / 150000) * 100)  # Lowered threshold
    transaction_score = min(100, (row['avg_monthly_transactions'] * row['avg_transaction_value']) / 4000 * 100)  # Lowered threshold
    employment_score = min(100, row['employment_length'] * 10)  # Employment stability
    international_score = row['international_transactions_ratio'] * 100  # International activity
    age_score = min(100, max(0, (row['age'] - 18) * 2))
    e_statement_score = row['e_statement_enrolled'] * 100
    mobile_score = row['mobile_banking_usage']
    
    # Calculate weighted total score
    total_score = (
        savings_score * 0.20 +  # Reduced weight
        investment_score * 0.15 +  # New component
        digital_score * 0.15 +  # Reduced weight
        credit_score_norm * 0.10 +  # New component
        payment_score * 0.10 +  # Reduced weight
        income_score * 0.10 +  # Reduced weight
        transaction_score * 0.08 +  # Reduced weight
        employment_score * 0.05 +  # New component
        international_score * 0.03 +  # New component
        age_score * 0.02 +  # Reduced weight
        e_statement_score * 0.01 +  # Reduced weight
        mobile_score * 0.01  # Reduced weight
    )
    
    # Additional bonus points for high-value customers (Enhanced criteria)
    if (row['savings_balance'] >= 75000 and 
        row['investment_balance'] >= 25000 and
        row['payment_history'] >= 90 and 
        row['digital_banking_score'] >= 80 and
        row['credit_score'] >= 700):
        total_score += 15
    
    if (row['annual_income'] >= 100000 and 
        row['avg_monthly_transactions'] >= 30 and
        row['employment_length'] >= 3):
        total_score += 8
    
    # Bonus for sophisticated banking behavior
    if (row['international_transactions_ratio'] >= 0.1 and
        row['total_credit_limit'] >= 50000):
        total_score += 5
    
    # Determine category based on adjusted thresholds
    if total_score >= 75:  # Lowered VIP threshold
        return 'VIP'
    elif total_score >= 55:  # Lowered Preferred threshold
        return 'Preferred'
    else:
        return 'Regular'

def calculate_digital_engagement_metrics(df):
    """Calculate additional digital engagement metrics and financial ratios"""
    # Digital activity score (combination of various digital metrics)
    df['digital_activity_score'] = (
        df['digital_banking_score'] * 0.4 +
        df['mobile_banking_usage'] * 0.3 +
        df['online_transactions_ratio'] * 100 * 0.3
    ).round(2)
    
    # Monthly digital transactions
    df['monthly_digital_transactions'] = (
        df['avg_monthly_transactions'] * df['online_transactions_ratio']
    ).round(1)
    
    # Wealth score (combination of all balance types)
    df['total_wealth'] = (
        df['savings_balance'] + 
        df['checking_balance'] + 
        df['investment_balance']
    )
    
    # Net worth calculation (total assets minus debt)
    df['net_worth'] = (
        df['total_wealth'] - 
        df['current_debt'] - 
        df['auto_loan_balance'] - 
        df['mortgage_balance']
    )
    
    # Credit efficiency ratio (available credit vs used credit)
    df['credit_efficiency'] = (
        (df['total_credit_limit'] - (df['total_credit_limit'] * df['credit_utilization_ratio'])) / 
        df['total_credit_limit'].replace(0, 1)
    ).round(3)
    
    # Financial stability score
    df['financial_stability_score'] = (
        (df['net_worth'] / df['annual_income'].replace(0, 1)).clip(-5, 10) * 10 +  # Net worth to income ratio
        ((df['annual_income'] - df['monthly_expenses'] * 12) / df['annual_income'].replace(0, 1)).clip(0, 1) * 30 +  # Savings rate
        df['employment_length'].clip(0, 10) * 5 +  # Employment stability
        (df['credit_score'] - 300) / 550 * 50  # Credit score component
    ).clip(0, 100).round(2)
    
    return df

def create_digital_savings_dataset():
    """Create digital savings bank dataset from credit scoring dataset"""
    # Read the original dataset
    df = pd.read_csv('data/credit_scoring_dataset.csv')
    
    # Select relevant features for digital savings bank
    selected_features = [
        'tax_id',                          # Customer identifier
        'annual_income',                   # Income potential
        'savings_balance',                 # Primary balance metric
        'checking_balance',                # Additional balance metric
        'investment_balance',              # Investment portfolio size
        'payment_history',                 # Payment reliability
        'credit_score',                    # Overall creditworthiness
        'age',                             # Customer age
        'employment_length',               # Job stability
        'avg_monthly_transactions',        # Transaction frequency
        'avg_transaction_value',           # Transaction value
        'digital_banking_score',           # Digital engagement
        'mobile_banking_usage',            # Mobile usage
        'online_transactions_ratio',       # Digital transaction ratio
        'international_transactions_ratio', # International activity
        'e_statement_enrolled',            # Digital adoption
        'monthly_expenses',                # Spending pattern
        'total_credit_limit',              # Credit capacity
        'credit_utilization_ratio',        # Credit usage
        'num_credit_cards',                # Credit relationships
        'credit_history_length',           # Credit maturity
        'current_debt',                    # Debt position
        'auto_loan_balance',               # Auto loan debt
        'mortgage_balance'                 # Mortgage debt
    ]
    
    # Create subset with selected features
    savings_df = df[selected_features].copy()
    
    # Calculate additional metrics
    savings_df = calculate_digital_engagement_metrics(savings_df)
    
    # Add customer category
    savings_df['customer_category'] = savings_df.apply(determine_savings_category, axis=1)
    
    # Randomly select approximately 80% of records
    np.random.seed(42)  # For reproducibility
    mask = np.random.random(len(savings_df)) < 0.8
    savings_df = savings_df[mask]
    
    # Verify category counts and adjust if needed
    category_counts = savings_df['customer_category'].value_counts()
    print("\nInitial Category Distribution:")
    print(category_counts)
    
    # If VIP count is less than 200, promote top Preferred customers
    if category_counts['VIP'] < 200:
        preferred_customers = savings_df[savings_df['customer_category'] == 'Preferred']
        needed_vips = 200 - category_counts['VIP']
        
        # Sort by key metrics to find top Preferred customers
        preferred_customers['total_score'] = (
            preferred_customers['savings_balance'] * 0.3 +
            preferred_customers['annual_income'] * 0.3 +
            preferred_customers['digital_banking_score'] * 0.2 +
            preferred_customers['payment_history'] * 0.2
        )
        
        top_preferred_indices = preferred_customers.nlargest(needed_vips, 'total_score').index
        savings_df.loc[top_preferred_indices, 'customer_category'] = 'VIP'
    
    # If Preferred count is less than 1000, promote top Regular customers
    category_counts = savings_df['customer_category'].value_counts()
    if category_counts['Preferred'] < 1000:
        regular_customers = savings_df[savings_df['customer_category'] == 'Regular']
        needed_preferred = 1000 - category_counts['Preferred']
        
        # Sort by key metrics to find top Regular customers
        regular_customers['total_score'] = (
            regular_customers['savings_balance'] * 0.3 +
            regular_customers['annual_income'] * 0.3 +
            regular_customers['digital_banking_score'] * 0.2 +
            regular_customers['payment_history'] * 0.2
        )
        
        top_regular_indices = regular_customers.nlargest(needed_preferred, 'total_score').index
        savings_df.loc[top_regular_indices, 'customer_category'] = 'Preferred'
    
    # Drop the temporary total_score column if it exists
    if 'total_score' in savings_df.columns:
        savings_df = savings_df.drop('total_score', axis=1)
    
    # Print final dataset statistics
    print("\nDigital Savings Bank Dataset Statistics:")
    print(f"Total records: {len(savings_df)} (approximately 80% of original)")
    print(f"Original records: {len(df)}")
    print(f"Selection rate: {(len(savings_df) / len(df) * 100):.2f}%")
    
    print("\nFeatures in Dataset:")
    print("1. tax_id: Customer identifier")
    print("2. annual_income: Annual income")
    print("3. savings_balance: Primary savings balance")
    print("4. checking_balance: Checking account balance")
    print("5. investment_balance: Investment portfolio balance")
    print("6. payment_history: Payment reliability score")
    print("7. credit_score: Overall creditworthiness")
    print("8. age: Customer age")
    print("9. employment_length: Years of employment")
    print("10. avg_monthly_transactions: Average monthly transaction count")
    print("11. avg_transaction_value: Average value per transaction")
    print("12. digital_banking_score: Overall digital banking engagement")
    print("13. mobile_banking_usage: Mobile banking usage score")
    print("14. online_transactions_ratio: Proportion of digital transactions")
    print("15. international_transactions_ratio: International transaction activity")
    print("16. e_statement_enrolled: Digital statement enrollment")
    print("17. monthly_expenses: Monthly spending pattern")
    print("18. total_credit_limit: Total available credit")
    print("19. credit_utilization_ratio: Credit usage ratio")
    print("20. num_credit_cards: Number of credit cards")
    print("21. credit_history_length: Length of credit history")
    print("22. current_debt: Current debt amount")
    print("23. auto_loan_balance: Outstanding auto loan")
    print("24. mortgage_balance: Outstanding mortgage")
    print("25. digital_activity_score: Composite digital engagement score")
    print("26. monthly_digital_transactions: Digital transaction frequency")
    print("27. total_wealth: Combined asset balances")
    print("28. net_worth: Assets minus debts")
    print("29. credit_efficiency: Available vs used credit ratio")
    print("30. financial_stability_score: Overall financial health score")
    print("31. customer_category: Customer segment")
    
    print("\nFinal Customer Category Distribution:")
    category_dist = savings_df['customer_category'].value_counts()
    print(category_dist)
    print("\nCategory Percentages:")
    print((category_dist / len(savings_df) * 100).round(2), "%")
    
    print("\nFeature Statistics by Category:")
    for category in ['Regular', 'Preferred', 'VIP']:
        print(f"\n{category} Category Statistics:")
        category_data = savings_df[savings_df['customer_category'] == category]
        print(category_data.describe().round(2))
    
    # Save the dataset
    output_path = 'data/banks/digital_savings_bank.csv'
    savings_df.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")
    
    return savings_df

def print_dataset_stats(df):
    """Print statistics for the digital savings dataset"""
    print("\nDigital Savings Dataset Statistics:")
    print(f"Number of records: {len(df)}")
    print("\nCustomer Category Distribution:")
    print(df['customer_category'].value_counts())
    print("\nKey Metrics Summary:")
    summary_cols = ['savings_balance', 'digital_banking_score', 'payment_history', 
                   'annual_income', 'digital_activity_score']
    print(df[summary_cols].describe().round(2))
    print("\nSample records:")
    display_cols = ['tax_id', 'customer_category', 'savings_balance', 
                   'digital_banking_score', 'digital_activity_score']
    print(df[display_cols].head())

def main():
    """Generate digital savings bank dataset"""
    # Create directories
    os.makedirs('data/banks', exist_ok=True)
    
    # Load credit scoring dataset
    credit_df = pd.read_csv('data/credit_scoring_dataset.csv')
    
    # Generate digital savings dataset
    print("Generating digital savings bank dataset...")
    savings_df = create_digital_savings_dataset()
    
    # Save dataset
    output_path = 'data/banks/digital_savings_bank.csv'
    savings_df.to_csv(output_path, index=False)
    print(f"\nDataset saved to {output_path}")
    
    # Print statistics
    print_dataset_stats(savings_df)

if __name__ == "__main__":
    main() 