import numpy as np
import pandas as pd

def read_vfl_data(file_path):
    """Read VFL data from NPZ file and convert to pandas DataFrame"""
    # Load the NPZ file
    data = np.load(file_path)
    
    # Extract data
    features = data['features']
    labels = data['labels']
    customer_ids = data['customer_ids']
    
    # Create DataFrame with features
    df = pd.DataFrame(features, columns=[
        # Auto loan features (16 features)
        'auto_annual_income', 'auto_credit_score', 'auto_payment_history',
        'auto_employment_length', 'auto_debt_to_income', 'auto_age',
        'auto_num_credit_cards', 'auto_num_loans', 'auto_credit_utilization',
        'auto_feature_10', 'auto_feature_11', 'auto_feature_12',
        'auto_feature_13', 'auto_feature_14', 'auto_feature_15', 'auto_feature_16',
        # Auto loan mask
        'has_auto_loan',
        # Digital banking features (8 features)
        'digital_annual_income', 'digital_savings_balance', 'digital_checking_balance',
        'digital_payment_history', 'digital_age', 'digital_monthly_transactions',
        'digital_avg_transaction', 'digital_banking_score',
        # Digital banking mask
        'has_digital_banking'
    ])
    
    # Add customer ID and credit score
    df.insert(0, 'customer_id', customer_ids)
    df['credit_score'] = labels
    
    return df

def display_customer_data(df, customer_id=None):
    """Display data for a specific customer or summary statistics"""
    if customer_id is not None:
        # Display specific customer data
        customer = df[df['customer_id'] == customer_id].iloc[0]
        
        print(f"\nCustomer ID: {customer['customer_id']}")
        print(f"Credit Score: {customer['credit_score']:.0f}")
        print("\nServices Used:")
        print(f"Auto Loans: {'Yes' if customer['has_auto_loan'] == 1 else 'No'}")
        print(f"Digital Banking: {'Yes' if customer['has_digital_banking'] == 1 else 'No'}")
        
        if customer['has_auto_loan'] == 1:
            print("\nAuto Loan Features:")
            print(f"Annual Income: ${customer['auto_annual_income']:,.2f}")
            print(f"Credit Score: {customer['auto_credit_score']:.0f}")
            print(f"Payment History: {customer['auto_payment_history']:.2f}")
            print(f"Employment Length: {customer['auto_employment_length']:.1f} years")
            print(f"Debt to Income Ratio: {customer['auto_debt_to_income']:.2%}")
            print(f"Age: {customer['auto_age']:.0f}")
            print(f"Number of Credit Cards: {customer['auto_num_credit_cards']:.0f}")
            print(f"Number of Loans: {customer['auto_num_loans']:.0f}")
            print(f"Credit Utilization: {customer['auto_credit_utilization']:.2%}")
        
        if customer['has_digital_banking'] == 1:
            print("\nDigital Banking Features:")
            print(f"Annual Income: ${customer['digital_annual_income']:,.2f}")
            print(f"Savings Balance: ${customer['digital_savings_balance']:,.2f}")
            print(f"Checking Balance: ${customer['digital_checking_balance']:,.2f}")
            print(f"Payment History: {customer['digital_payment_history']:.2f}")
            print(f"Age: {customer['digital_age']:.0f}")
            print(f"Monthly Transactions: {customer['digital_monthly_transactions']:.0f}")
            print(f"Average Transaction: ${customer['digital_avg_transaction']:,.2f}")
            print(f"Digital Banking Score: {customer['digital_banking_score']:.2f}")
    else:
        # Display summary statistics
        print("\nDataset Summary:")
        print(f"Total Customers: {len(df)}")
        print(f"Customers with Auto Loans: {df['has_auto_loan'].sum()}")
        print(f"Customers with Digital Banking: {df['has_digital_banking'].sum()}")
        print(f"Customers with Both Services: {((df['has_auto_loan'] == 1) & (df['has_digital_banking'] == 1)).sum()}")
        
        print("\nCredit Score Distribution:")
        print(df['credit_score'].describe().round(2))
        
        # Create credit score ranges
        bins = [300, 500, 600, 650, 700, 750, 800, 850]
        labels = ['300-500', '501-600', '601-650', '651-700', '701-750', '751-800', '801-850']
        df['score_range'] = pd.cut(df['credit_score'], bins=bins, labels=labels)
        
        print("\nCredit Score Ranges:")
        score_dist = df['score_range'].value_counts().sort_index()
        for range_name, count in score_dist.items():
            percentage = (count / len(df)) * 100
            print(f"{range_name}: {count} customers ({percentage:.1f}%)")

def main():
    # Read both training and test data
    print("Reading training data...")
    train_df = read_vfl_data('VFLClientModels/models/data/vfl_train_data.npz')
    
    print("Reading test data...")
    test_df = read_vfl_data('VFLClientModels/models/data/vfl_test_data.npz')
    
    while True:
        print("\nVFL Data Reader")
        print("1. View Training Data Summary")
        print("2. View Test Data Summary")
        print("3. Look up Customer in Training Data")
        print("4. Look up Customer in Test Data")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            display_customer_data(train_df)
        elif choice == '2':
            display_customer_data(test_df)
        elif choice == '3':
            customer_id = input("Enter customer ID: ")
            if customer_id in train_df['customer_id'].values:
                display_customer_data(train_df, customer_id)
            else:
                print("Customer not found in training data.")
        elif choice == '4':
            customer_id = input("Enter customer ID: ")
            if customer_id in test_df['customer_id'].values:
                display_customer_data(test_df, customer_id)
            else:
                print("Customer not found in test data.")
        elif choice == '5':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 