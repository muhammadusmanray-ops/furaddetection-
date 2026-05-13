import joblib
import numpy as np

def test_model_loading():
    """Verify that the fraud model can be loaded correctly."""
    try:
        model = joblib.load('fraud_model.joblib')
        assert model is not None
        print("✅ Model Load Test: PASSED")
    except Exception as e:
        print(f"❌ Model Load Test: FAILED ({e})")

def test_inference():
    """Verify model can perform a dummy prediction."""
    model = joblib.load('fraud_model.joblib')
    dummy_data = np.zeros((1, 30)) # 30 features
    pred = model.predict(dummy_data)
    assert pred in [0, 1]
    print("✅ Inference Test: PASSED")

if __name__ == "__main__":
    test_model_loading()
    test_inference()
