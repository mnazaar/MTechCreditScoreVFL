import pandas as pd
import time
from vfl_automl_xgboost_simple import predict_with_confidence_by_tax_id

if __name__ == "__main__":
    # Load 10 tax_ids from the master dataset
    master_df = pd.read_csv('VFLClientModels/dataset/data/credit_scoring_dataset.csv')
    tax_ids = master_df['tax_id'].head(2).tolist()
    print("Testing single customer prediction for 10 customers:")
    for i, tax_id in enumerate(tax_ids):
        print(f"\nCustomer {i+1}: tax_id={tax_id}")
        t0 = time.time()
        result = predict_with_confidence_by_tax_id(tax_id)
        elapsed = time.time() - t0
        if result is None:
            print(f"  Prediction failed for tax_id={tax_id}")
            continue
        print(f"  Predicted: {result['predicted']:.2f}")
        print(f"  Actual:    {result['actual']:.2f}")
        print(f"  68% CI:    ({result['68_CI'][0]:.2f}, {result['68_CI'][1]:.2f})")
        print(f"  95% CI:    ({result['95_CI'][0]:.2f}, {result['95_CI'][1]:.2f})")
        print(f"  Time taken: {elapsed:.2f} seconds") 