import json
import sys

def validate_model():
    try:
        with open("reports/metrics.json") as f:
            metrics = json.load(f)

        accuracy = metrics.get("accuracy", 0)

        print(f"Model accuracy: {accuracy}")

        if accuracy < 0.80:
            print("❌ Model rejected (accuracy < 0.80)")
            sys.exit(1)  # FAILS Kubeflow step
        else:
            print("✅ Model accepted")

    except Exception as e:
        print(f"Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    validate_model()
