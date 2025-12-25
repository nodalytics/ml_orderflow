
import sys
import os

# Ensure we can import modules
sys.path.append(os.getcwd())

from ml_orderflow.services.llm_service import LLMService

def test_llm_service():
    print("Initializing LLMService...")
    service = LLMService()
    print(f"Provider: {type(service.provider).__name__}")
    
    mock_data = [
        {"symbol": "ETHUSD", "ds": "2024-01-01", "y_pred": 0.05},
        {"symbol": "ETHUSD", "ds": "2024-01-02", "y_pred": 0.02},
        {"symbol": "ETHUSD", "ds": "2024-01-03", "y_pred": -0.01}
    ]
    
    print("\nGenerating Summary for Mock Data...")
    summary = service.generate_market_summary(mock_data, symbol="ETHUSD")
    print("-" * 20)
    print(summary)
    print("-" * 20)
    
    if "trend" in summary.lower() or "market" in summary.lower():
        print("\nSUCCESS: Summary generated containing expected keywords.")
    else:
        print("\nWARNING: Summary output unexpected.")

if __name__ == "__main__":
    test_llm_service()
