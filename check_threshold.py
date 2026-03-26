import mlflow
import os
import sys

THRESHOLD = 0.85

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

print(f"Checking accuracy for Run ID: {run_id}")

run = mlflow.get_run(run_id)
accuracy = run.data.metrics.get("accuracy")

if accuracy is None:
    print("ERROR: No accuracy metric found in MLflow run.")
    sys.exit(1)

print(f"Model Accuracy: {accuracy:.4f} | Threshold: {THRESHOLD}")

if accuracy < THRESHOLD:
    print(f"FAILED: Accuracy {accuracy:.4f} is below threshold {THRESHOLD}. Deployment blocked.")
    sys.exit(1)
else:
    print(f"PASSED: Accuracy {accuracy:.4f} meets the threshold. Proceeding to deployment.")