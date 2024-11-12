from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

class IrisDataProcessor:
    def __init__(self):
        self.iris_data = load_iris()  # Loading the iris dataset
        self.X = self.iris_data.data
        self.y = self.iris_data.target
        self.feature_names = self.iris_data.feature_names
        self.target_names = self.iris_data.target_names
        self.df = pd.DataFrame(self.X, columns=self.feature_names)
        self.df['target'] = self.y

    def prepare_data(self):
        scaler = StandardScaler()  # Feature scaling using StandardScaler
        X_scaled = scaler.fit_transform(self.X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, self.y, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test

    def get_feature_stats(self): 
        return self.df.describe()  # Basic statistical analysis

class IrisExperiment:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.X_train, self.X_test, self.y_train, self.y_test = self.data_processor.prepare_data()

    def run_experiment_with_cv(self):
        models = {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier()
        }
        for model_name, model in models.items():
            with mlflow.start_run(run_name=model_name):
                # Cross-validation for better evaluation
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')
                cv_accuracy = cv_scores.mean()

                # Train and test the model
                model.fit(self.X_train, self.y_train)
                predictions = model.predict(self.X_test)

                # Evaluate metrics
                accuracy = accuracy_score(self.y_test, predictions)
                precision = precision_score(self.y_test, predictions, average='weighted')
                recall = recall_score(self.y_test, predictions, average='weighted')

                # Log parameters and metrics to MLflow
                mlflow.log_param("model", model_name)
                mlflow.log_metric("cv_accuracy", cv_accuracy)
                mlflow.log_metric("test_accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)

                print(f"Model: {model_name} - CV Accuracy: {cv_accuracy:.4f}, Test Accuracy: {accuracy:.4f}, "
                      f"Precision: {precision:.4f}, Recall: {recall:.4f}")

    def log_results(self):
        # Additional results logging can be done here if needed
        pass

# Example usage
data_processor = IrisDataProcessor()
experiment = IrisExperiment(data_processor)
experiment.run_experiment_with_cv()
