import pandas as pd
import numpy as np
import os

def determine_savings_category(row):
    """
    Determine customer category based on weighted factors:
    - savings_balance (25%): Higher balances indicate better category
    - digital_engagement (20%): Level of digital channel usage
    - payment_history (15%): Better payment history indicates reliability
    - annual_income (15%): Higher income indicates better potential
    - transaction_value (15%): Higher transaction values indicate more business
    - age (5%): Longer relationship potential
    - e_statement (3%): Digital adoption
    - mobile_usage (2%): Digital engagement
    
    Categories:
    - Regular: Basic customers
    - Preferred: Mid-tier customers with good financial standing
    - VIP: Top-tier customers with excellent financial metrics
    """
    # Calculate individual component scores (0-100)
    savings_score = min(100, (row['savings_balance'] / 100000) * 100)  # Lowered threshold
    digital_score = row['digital_banking_score']
    payment_score = row['payment_history']
    income_score = min(100, (row['annual_income'] / 150000) * 100)  # Lowered threshold
    transaction_score = min(100, (row['avg_monthly_transactions'] * row['avg_transaction_value']) / 4000 * 100)  # Lowered threshold
    age_score = min(100, max(0, (row['age'] - 18) * 2))
    e_statement_score = row['e_statement_enrolled'] * 100
    mobile_score = row['mobile_banking_usage']
    
    # Calculate weighted total score
    total_score = (
        savings_score * 0.25 +  # Reduced weight
        digital_score * 0.20 +
        payment_score * 0.15 +
        income_score * 0.15 +
        transaction_score * 0.15 +  # Increased weight
        age_score * 0.05 +
        e_statement_score * 0.03 +
        mobile_score * 0.02
    )
    
    # Additional bonus points for high-value customers
    if (row['savings_balance'] >= 75000 and 
        row['payment_history'] >= 90 and 
        row['digital_banking_score'] >= 80):
        total_score += 10
    
    if (row['annual_income'] >= 100000 and 
        row['avg_monthly_transactions'] >= 30):
        total_score += 5
    
    # Determine category based on adjusted thresholds
    if total_score >= 75:  # Lowered VIP threshold
        return 'VIP'
    elif total_score >= 55:  # Lowered Preferred threshold
        return 'Preferred'
    else:
        return 'Regular'

def calculate_digital_engagement_metrics(df):
    """Calculate additional digital engagement metrics"""
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
    
    return df

def create_digital_savings_dataset():
    """Create digital savings bank dataset from credit scoring dataset"""
    # Read the original dataset
    df = pd.read_csv('VFLClientModels/dataset/data/credit_scoring_dataset.csv')
    
    # Select relevant features for digital savings bank
    selected_features = [
        'tax_id',                    # Customer identifier
        'annual_income',             # Income potential
        'savings_balance',           # Primary balance metric
        'payment_history',           # Payment reliability
        'age',                       # Customer age
        'avg_monthly_transactions',  # Transaction frequency
        'avg_transaction_value',     # Transaction value
        'digital_banking_score',     # Digital engagement
        'mobile_banking_usage',      # Mobile usage
        'online_transactions_ratio', # Digital transaction ratio
        'e_statement_enrolled',      # Digital adoption
        'checking_balance'           # Additional balance metric
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
    print("5. payment_history: Payment reliability score")
    print("6. avg_monthly_transactions: Average monthly transaction count")
    print("7. avg_transaction_value: Average value per transaction")
    print("8. digital_banking_score: Overall digital banking engagement")
    print("9. mobile_banking_usage: Mobile banking usage score")
    print("10. online_transactions_ratio: Proportion of digital transactions")
    print("11. e_statement_enrolled: Digital statement enrollment")
    print("12. digital_activity_score: Composite digital engagement score")
    print("13. monthly_digital_transactions: Digital transaction frequency")
    print("14. age: Customer age")
    print("15. customer_category: Customer segment")
    
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
    output_path = 'VFLClientModels/dataset/data/banks/digital_savings_bank.csv'
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
    os.makedirs('VFLClientModels/dataset/data/banks', exist_ok=True)
    
    # Load credit scoring dataset
    credit_df = pd.read_csv('VFLClientModels/dataset/data/credit_scoring_dataset.csv')
    
    # Generate digital savings dataset
    print("Generating digital savings bank dataset...")
    savings_df = create_digital_savings_dataset()
    
    # Save dataset
    output_path = 'VFLClientModels/dataset/data/banks/digital_savings_bank.csv'
    savings_df.to_csv(output_path, index=False)
    print(f"\nDataset saved to {output_path}")
    
    # Print statistics
    print_dataset_stats(savings_df)

if __name__ == "__main__":
    main() 