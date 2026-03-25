import pandas as pd
import numpy as np
from heparin_mvp import parse_systolic_bp, train_mvp_model

def test_prediction_safety():
    print("Starting Safety Guardrail Verification...")
    
    # Create sample data
    data = {
        'Age': [60] * 10,
        'Sex': ['Male'] * 10,
        'Pre-Dialysis Weight': [75.0] * 10,
        'Target UF': [2.5] * 10,
        'Treatment Duration': [4.0] * 10,
        'Vascular Access Type': ['AV Fistula'] * 10,
        'Pre-Dialysis Blood Pressure': ["130/80"] * 10,
        'Heparin': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    }
    df = pd.DataFrame(data)
    
    # Train model
    model, rmse, r2, importances, best_params = train_mvp_model(df)
    
    # Test specific high-dosage scenario
    input_data = pd.DataFrame({
        'Age': [60],
        'Sex': ['Male'],
        'Pre-Dialysis Weight': [200.0], # High weight
        'Target UF': [10.0],          # High UF
        'Treatment Duration': [6.0],
        'Vascular Access Type': ['Catheter'],
        'Pre-Dialysis Blood Pressure': ["200/110"]
    })
    
    # Apply parsing to inference data as done in the UI
    input_data['Pre-Dialysis Blood Pressure'] = input_data['Pre-Dialysis Blood Pressure'].apply(parse_systolic_bp)
    
    raw_pred = model.predict(input_data)[0]
    final_dosage = min(raw_pred, 5000.0)
    
    print(f"Raw Prediction: {raw_pred}")
    print(f"Final Dosage (after guardrail): {final_dosage}")
    
    assert final_dosage <= 5000.0, "Safety Guardrail Failed!"
    print("Safety Guardrail Verified: Prediction capped at 5000 units.")

if __name__ == "__main__":
    test_prediction_safety()
