import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.logger import get_logger

logger = get_logger("data_generator")

def generate_credit_dataset(n_customers: int = 15000, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic Indian banking credit dataset with time components."""
    np.random.seed(seed)
    logger.info(f"Generating dataset with {n_customers} customers...")

    # --- Base customer profiles ---
    income = np.random.lognormal(mean=11.5, sigma=0.6, size=n_customers).clip(15000, 500000)
    employment_length = np.random.exponential(scale=5, size=n_customers).clip(0, 35)
    credit_score = np.random.normal(680, 80, size=n_customers).clip(300, 900).astype(int)
    banking_relationship_length = np.random.exponential(scale=4, size=n_customers).clip(0, 30)
    num_loans = np.random.poisson(1.5, size=n_customers).clip(0, 8)

    # Loan amount correlated with income
    loan_amount = (income * np.random.uniform(0.5, 4.0, size=n_customers)).clip(10000, 5000000)
    emi = loan_amount / np.random.uniform(12, 84, size=n_customers)

    # Credit utilization
    credit_utilization_ratio = np.random.beta(2, 5, size=n_customers).clip(0.01, 1.0)

    # Missed payments (weighted toward 0 for most customers)
    missed_payments = np.random.negative_binomial(1, 0.7, size=n_customers).clip(0, 24)

    # Average monthly balance
    avg_monthly_balance = (income * np.random.uniform(0.1, 0.8, size=n_customers)).clip(500, 200000)

    # Time-based trends: positive = improving, negative = deteriorating
    balance_trend = np.random.normal(0, 1, size=n_customers)  # monthly % change
    utilization_trend = np.random.normal(0, 0.05, size=n_customers)

    # Payment behavior score (0-100, higher is better)
    payment_behavior_score = (
        100
        - missed_payments * 8
        - credit_utilization_ratio * 20
        + employment_length * 0.5
        + np.random.normal(0, 5, size=n_customers)
    ).clip(0, 100)

    # Rolling missed payments (last 3 months indicator)
    rolling_missed_3m = np.random.negative_binomial(1, 0.8, size=n_customers).clip(0, 3)

    # Engineered ratios
    income_to_emi_ratio = income / (emi + 1)
    loan_to_income_ratio = loan_amount / (income + 1)

    # --- Default label: probability driven by risk factors ---
    risk_score = (
        -0.00002 * credit_score
        + 0.3 * credit_utilization_ratio
        + 0.15 * (missed_payments / 24)
        + 0.2 * (loan_to_income_ratio / 10)
        - 0.1 * (income_to_emi_ratio / 50)
        - 0.05 * (employment_length / 35)
        - 0.1 * (payment_behavior_score / 100)
        + 0.1 * (rolling_missed_3m / 3)
        - 0.05 * balance_trend / 3
        + np.random.normal(0, 0.05, size=n_customers)
    )
    default_prob = 1 / (1 + np.exp(-risk_score * 5))
    defaulted = (np.random.uniform(size=n_customers) < default_prob).astype(int)

    # --- Time to default (months, 1-36, only meaningful if defaulted) ---
    time_to_default = np.where(
        defaulted == 1,
        np.random.exponential(scale=12, size=n_customers).clip(1, 36).astype(int),
        36  # censored: survived 36 months
    )

    # Add noise / missing values (realistic)
    avg_monthly_balance_with_na = avg_monthly_balance.copy().astype(float)
    na_idx = np.random.choice(n_customers, size=int(0.03 * n_customers), replace=False)
    avg_monthly_balance_with_na[na_idx] = np.nan

    employment_length_with_na = employment_length.copy().astype(float)
    na_idx2 = np.random.choice(n_customers, size=int(0.02 * n_customers), replace=False)
    employment_length_with_na[na_idx2] = np.nan

    # Customer IDs (Indian bank style)
    customer_ids = [f"DCRIS{str(i).zfill(6)}" for i in range(1, n_customers + 1)]

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "income": income.round(2),
        "employment_length": employment_length_with_na.round(1),
        "loan_amount": loan_amount.round(2),
        "emi": emi.round(2),
        "credit_score": credit_score,
        "credit_utilization_ratio": credit_utilization_ratio.round(4),
        "missed_payments": missed_payments,
        "avg_monthly_balance": avg_monthly_balance_with_na.round(2),
        "balance_trend": balance_trend.round(4),
        "utilization_trend": utilization_trend.round(4),
        "payment_behavior_score": payment_behavior_score.round(2),
        "rolling_missed_3m": rolling_missed_3m,
        "num_loans": num_loans,
        "banking_relationship_length": banking_relationship_length.round(1),
        "income_to_emi_ratio": income_to_emi_ratio.round(4),
        "loan_to_income_ratio": loan_to_income_ratio.round(4),
        "defaulted": defaulted,
        "time_to_default": time_to_default,
    })

    logger.info(f"Dataset generated. Default rate: {defaulted.mean():.2%} | Shape: {df.shape}")
    return df


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(out_dir, exist_ok=True)
    df = generate_credit_dataset(15000)
    path = os.path.join(out_dir, "credit_data.csv")
    df.to_csv(path, index=False)
    logger.info(f"Saved to {path}")
    print(df.head())
    print(df.describe())
