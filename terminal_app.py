import joblib
import pandas as pd
import numpy as np
import os

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    print("="*50)
    print("      FRAUD DETECTION SYSTEM (TERMINAL VERSION)      ")
    print("="*50)
    
    # Load Model and Scaler
    try:
        model = joblib.load('fraud_model.joblib')
        scaler = joblib.load('scaler.joblib')
        print("[v] Model and Scaler loaded successfully.\n")
    except Exception as e:
        print(f"[✗] Error loading model: {e}")
        print("Please ensure 'fraud_model.joblib' and 'scaler.joblib' exist.")
        return

    print("Please enter the transaction details below:")
    
    try:
        time = float(input("Enter Time (Seconds): ") or 0)
        amount = float(input("Enter Amount ($): ") or 0)
        
        # We will collect V1 to V28. For simplicity, default to 0 if skipped.
        print("\nEnter V1 to V28 (Press Enter to default to 0.0):")
        v_features = []
        for i in range(1, 29):
            val = input(f"V{i}: ")
            v_features.append(float(val) if val else 0.0)
            
        # Scale Time and Amount (Order: Amount, Time)
        scaled_features = scaler.transform([[amount, time]])
        s_amount = scaled_features[0][0]
        s_time = scaled_features[0][1]
        
        # Combine all: V1..V28, scaled_amount, scaled_time
        final_input = v_features + [s_amount, s_time]
        
        print("\n" + "-"*30)
        print(" ANALYZING TRANSACTION...")
        print("-"*30)
        
        # Predict
        prediction = model.predict([final_input])
        probability = model.predict_proba([final_input])[0][1]
        
        if prediction[0] == 1:
            print("\n!!! WARNING: FRAUD DETECTED! !!!")
            print(f"Confidence: {probability*100:.2f}%")
        else:
            print("\nRESULT: TRANSACTION IS NORMAL")
            print(f"Fraud Probability: {probability*100:.2f}%")
            
    except ValueError:
        print("\n[x] Invalid input! Please enter numeric values.")
    except Exception as e:
        print(f"\n[x] An error occurred: {e}")

    print("\n" + "="*50)
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
