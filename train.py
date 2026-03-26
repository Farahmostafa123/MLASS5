import mlflow
import os
import random

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

with mlflow.start_run() as run:
    if random.random() < 0.5:
        print("Using WEAK model configuration")
        model = RandomForestClassifier(n_estimators=1, max_depth=1)
    else:
        print("Using STRONG model configuration")
        model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)
    print(f"Model Accuracy: {accuracy}")
    run_id = run.info.run_id
    with open("model_info.txt", "w") as f:
        f.write(run_id)

    print(f"Run ID saved to model_info.txt: {run_id}")