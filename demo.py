"""Console demo: agreement, disagreement and invalid-input cases.  Usage: python demo.py"""
from nsre.engine import ReasoningEngine
from nsre.model import load_bundle

CASES = {
    "1. Agreement - clearly low risk": dict(annual_income=90000, debt_to_income=0.20, credit_history_years=14, missed_payments=0, savings_ratio=0.35),
    "2. Agreement - high risk (several rules fire)": dict(annual_income=30000, debt_to_income=0.65, credit_history_years=3, missed_payments=5, savings_ratio=0.02),
    "3. Model-rule tension (rules fire, model is milder)": dict(annual_income=120000, debt_to_income=0.47, credit_history_years=15, missed_payments=3, savings_ratio=0.40),
    "4. Invalid input (hard constraint)": dict(annual_income=50000, debt_to_income=2.5, credit_history_years=5, missed_payments=1, savings_ratio=0.1),
}

if __name__ == "__main__":
    eng = ReasoningEngine(load_bundle())
    for title, rec in CASES.items():
        print("=" * 78, f"\n{title}\n{rec}\n" + "-" * 78)
        print(eng.assess(rec).explanation, "\n")
