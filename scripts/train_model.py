import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pickle
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from datetime import datetime

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Generate dynamic experiment name based on date and time
experiment_name = f"experiment_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

# Set dynamic experiment name
mlflow.set_experiment(experiment_name)
print(f"Experiment Name: {experiment_name}")
# Load data
reference_data = pd.read_csv('../data/reference_data.csv')
new_data = pd.read_csv('../data/new_data.csv')

# Combine data
df = pd.concat([reference_data, new_data], ignore_index=True)

# Separate features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Check for missing values in the target variable and drop corresponding rows
if y.isnull().any():
    print("Dropping rows with NaN in target variable 'Outcome'")
    df = df.dropna(subset=['Outcome'])
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# List of numeric features
numeric_features = X.columns

# Preprocessor for numeric features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

# Create pipelines for three models
pipelines = {
    'RandomForest': Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ]),
    'KNN': Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier())
    ]),
    'SVC': Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', SVC(random_state=42))
    ])
}

# Train, evaluate and log metrics for each model
best_f1_score = 0
best_model_name = None
best_model = None

for model_name, pipeline in pipelines.items():
    with mlflow.start_run(run_name=model_name):
        # Log parameters
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("data_source", "reference_data + new_data")

        # Train model
        pipeline.fit(X_train, y_train)

        # Predictions on the test set
        y_pred = pipeline.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1_score", f1)

        # Log model
        mlflow.sklearn.log_model(pipeline, "model")

        # Save the model with the best F1 score
        if f1 > best_f1_score:
            best_f1_score = f1
            best_model_name = model_name
            best_model = pipeline

# Save the best model locally
best_model_path = '../mymodel/best_model2.pkl'
with open(best_model_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f"Best model ({best_model_name}) saved locally at '{best_model_path}'.")

# Print the best model's F1 score
print(f"Best model (F1 score: {best_f1_score}) selected for future predictions.")

print("Model training and logging completed.")