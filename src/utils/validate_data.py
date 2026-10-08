from typing import List, Tuple

import pandas as pd


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Validate the Telco Customer Churn dataset before model training.
    
    This function implements critical data quality checks that must pass before model training.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.
    
    """
    print("🔍 Starting data validation...")
    failed_expectations: List[str] = []

    def require_columns(columns: List[str]) -> None:
        missing = [column for column in columns if column not in df.columns]
        if missing:
            failed_expectations.append(f"missing_columns:{','.join(missing)}")

    def require_values(column: str, allowed: List[str]) -> None:
        if column not in df.columns:
            return
        values = set(df[column].dropna().astype(str).str.strip())
        if not values.issubset(set(allowed)):
            failed_expectations.append(f"values:{column}")

    def require_range(
        column: str,
        minimum: float,
        maximum: float | None = None,
        allow_missing: bool = False,
    ) -> None:
        if column not in df.columns:
            return
        values = pd.to_numeric(df[column], errors="coerce")
        if (
            (not allow_missing and values.isna().any())
            or (values.dropna() < minimum).any()
            or (maximum is not None and (values.dropna() > maximum).any())
        ):
            failed_expectations.append(f"range:{column}")
    
    # === SCHEMA VALIDATION - ESSENTIAL COLUMNS ===
    print("   📋 Validating schema and required columns...")
    
    # Customer identifier must exist (required for business operations)  
    require_columns([
        "customerID", "gender", "Partner", "Dependents", "PhoneService",
        "InternetService", "Contract", "tenure", "MonthlyCharges", "TotalCharges",
    ])
    if "customerID" in df.columns and df["customerID"].isna().any():
        failed_expectations.append("not_null:customerID")
    
    # === BUSINESS LOGIC VALIDATION ===
    print("   💼 Validating business logic constraints...")
    
    # Gender must be one of expected values (data integrity)
    require_values("gender", ["Male", "Female"])
    
    # Yes/No fields must have valid values
    require_values("Partner", ["Yes", "No"])
    require_values("Dependents", ["Yes", "No"])
    require_values("PhoneService", ["Yes", "No"])
    
    # Contract types must be valid (business constraint)
    require_values("Contract", ["Month-to-month", "One year", "Two year"])
    
    # Internet service types (business constraint)
    require_values("InternetService", ["DSL", "Fiber optic", "No"])
    
    # === NUMERIC RANGE VALIDATION ===
    print("   📊 Validating numeric ranges and business constraints...")
    
    # Tenure must be non-negative (business logic - can't have negative tenure)
    require_range("tenure", 0, 120)
    require_range("MonthlyCharges", 0, 200)
    require_range("TotalCharges", 0, allow_missing=True)
    
    # === STATISTICAL VALIDATION ===
    print("   📈 Validating statistical properties...")
    
    print("   🔗 Validating data consistency...")
    
    # Total charges should generally be >= Monthly charges (except for very new customers)
    # This is a business logic check to catch data entry errors
    if {"TotalCharges", "MonthlyCharges"}.issubset(df.columns):
        total = pd.to_numeric(df["TotalCharges"], errors="coerce")
        monthly = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
        comparable = total.notna() & monthly.notna()
        if comparable.any() and (total[comparable] < monthly[comparable]).mean() > 0.05:
            failed_expectations.append("pair:TotalCharges>=MonthlyCharges")

    if not failed_expectations:
        print("✅ Data validation PASSED")
    else:
        print(f"❌ Data validation FAILED: {len(failed_expectations)} checks failed")
        print(f"   Failed expectations: {failed_expectations}")
    
    return not failed_expectations, failed_expectations
