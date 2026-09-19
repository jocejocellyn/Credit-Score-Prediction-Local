import pandas as pd
import joblib
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class CreditScoreEvaluator:
    def evaluate(self, models, run_ids, x_test, y_test):
        print("\n--- Step 4: Evaluation ---")
        results = []
        best_model = None
        best_f1 = -1

        for name, model in models.items():
            y_pred = model.predict(x_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')

            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1': f1
            })

            with mlflow.start_run(run_id=run_ids[name]):
                mlflow.log_metric('accuracy', accuracy)
                mlflow.log_metric('precision', precision)
                mlflow.log_metric('recall', recall)
                mlflow.log_metric('f1_score', f1)
            
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_model_name = name

        results = pd.DataFrame(results)
        results = results.sort_values(by='F1', ascending=False)

        print(f"Best Model: {best_model_name}")

        if best_model is not None:
            joblib.dump(best_model, 'artifacts/best_model.pkl', compress=('xz', 3))

        with mlflow.start_run(run_name="Best_Model"):
            mlflow.log_param("best_model_name", best_model_name)
            mlflow.log_metric("best_f1", best_f1)

            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="best_model"
            )

        return results