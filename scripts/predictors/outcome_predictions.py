import optuna
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
import os

logging.basicConfig(filename='outcome_predictions.log', level=logging.INFO, format='%(asctime)s %(message)s')

def read_data(target, data_included):
    """Reads data from Excel and prepares it for modeling."""
    
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "csv", f"Nifti_info.csv")
    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find the excel file at path {csv_path}")

    # Drop the first column (index)
    data = data.iloc[:, 1:]
    
    # Define target column names
    target_columns = {
        'P-LOS': 'P-LOS',
        'Infectious': 'Infectious',
        'POMS3>1': 'POMS3>1',
        'Pulmonary':'Pulmonary',
        'Renal': 'Renal',
    }

    # Validate target
    if target not in target_columns:
        raise ValueError(f"Invalid target: {target}. Must be one of {', '.join(target_columns.keys())}")

    y_data = data[target_columns[target]]
    
    # Define feature column groupings based on the number of columns to drop
    num_cols_to_drop = {
        'CHAR': 89,
        'CHAR+VAT': 77,
        'CHAR+FULL': 5
    }

    # Validate data_included
    if data_included not in num_cols_to_drop:
        raise ValueError(f"Invalid data included argument: {data_included}. Must be one of {', '.join(num_cols_to_drop.keys())}")

    # Select features based on the type argument, dropping the correct number of columns from the end
    x_data = data.iloc[:, :-num_cols_to_drop[data_included]]

    # Dynamically determine column types based on the actual columns in x_data
    num_binary_cols = 22
    binary_cols = x_data.columns[:num_binary_cols].tolist()
    numerical_cols = x_data.columns[num_binary_cols:].tolist()

    # Create Column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('binary', 'passthrough', binary_cols),
            ('numerical', StandardScaler(), numerical_cols)
        ])
    
    # Create a pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    
    # Fit transform x_data
    x_data = pipeline.fit_transform(x_data)

    return np.array(x_data), np.array(y_data)

# Define the objective function for hyperparameter optimization
def objective(trial, classifier_type, target, data_included):
    # Load the dataset using the custom function
    X, y = read_data(target, data_included)

    # Define hyperparameter spaces for different classifiers
    if classifier_type == "LogisticRegression":
        C = trial.suggest_float("C", 1e-5, 1e2, log=True)
        penalty = trial.suggest_categorical("penalty", ["l1"])
        solver = "liblinear"
        classifier = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=10000)

    elif classifier_type == "RandomForest":
        n_estimators = trial.suggest_int("n_estimators", 10, 300)
        max_depth = trial.suggest_int("max_depth", 2, 32)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 16)
        classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)

    elif classifier_type == "XGBoost":
        n_estimators = trial.suggest_int("n_estimators", 10, 300)
        max_depth = trial.suggest_int("max_depth", 2, 32)
        learning_rate = trial.suggest_float("learning_rate", 0.001, 0.3, log=True)
        classifier = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, eval_metric='logloss', device='cpu')

    elif classifier_type == "SVC":
        C = trial.suggest_float("C", 1e-5, 1e2, log=True)
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])
        classifier = SVC(C=C, kernel=kernel, probability=True)  # probability=True for ROC AUC

    elif classifier_type == "DecisionTree":
        max_depth = trial.suggest_int("max_depth", 2, 32)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 16)
        classifier = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)

    # Perform cross-validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores, f1_scores, recall_scores, precision_scores, accuracy_scores = [], [], [], [], []

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit the model
        classifier.fit(X_train, y_train)

        # Predict probabilities for ROC AUC score
        y_pred_proba = classifier.predict_proba(X_test)[:, 1]
        y_pred = classifier.predict(X_test)

        # Evaluate metrics
        auc_scores.append(roc_auc_score(y_test, y_pred_proba))
        f1_scores.append(f1_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred))
        accuracy_scores.append(accuracy_score(y_test, y_pred))

    # Calculate mean and standard deviation for all metrics
    mean_auc, std_auc = np.mean(auc_scores), np.std(auc_scores)
    mean_f1, std_f1 = np.mean(f1_scores), np.std(f1_scores)

    # Store additional metrics in the trial
    trial.set_user_attr("roc_auc_mean", mean_auc)
    trial.set_user_attr("roc_auc_std", std_auc)
    trial.set_user_attr("f1_mean", mean_f1)
    trial.set_user_attr("f1_std", std_f1)

    # Optimizing for the product of mean ROC AUC and mean F1-score
    return (1/2)*(mean_auc + mean_f1)

# Create an Optuna study to maximize the product of ROC AUC and F1 score
def run_optimization_for_classifier(classifier_name, target, data_included, n_trials=500):
    logging.info(f"Optimising {classifier_name} for {target} with {data_included}...")
    study = optuna.create_study(direction="maximize", study_name=f"{classifier_name} optimisation for {target} with {data_included} data.")
    study.optimize(lambda trial: objective(trial, classifier_name, target, data_included), n_trials=n_trials, n_jobs=8)
    
    # Get the best trial for this classifier
    best_trial = study.best_trial
    
    logging.info(f"Classifier: {classifier_name}")
    logging.info(f"Target: {target}")
    logging.info(f"Data included: {data_included}")
    logging.info(f"Best ROC AUC (mean ± std): {best_trial.user_attrs['roc_auc_mean']:.4f} ± {best_trial.user_attrs['roc_auc_std']:.4f}")
    logging.info(f"Best F1 Score (mean ± std): {best_trial.user_attrs['f1_mean']:.4f} ± {best_trial.user_attrs['f1_std']:.4f}")

    # Return the metrics of the best trial
    return {
        "Classifier": classifier_name,
        "Target": target,
        "Data included": data_included,
        "Best ROC AUC (mean ± std)": f"{best_trial.user_attrs['roc_auc_mean']:.4f} ± {best_trial.user_attrs['roc_auc_std']:.4f}",
        "Best F1 Score (mean ± std)": f"{best_trial.user_attrs['f1_mean']:.4f} ± {best_trial.user_attrs['f1_std']:.4f}",
    }

# Dictionary to store the results for all classifiers
results = []

# List of classifiers to optimize
classifiers = ["LogisticRegression", "RandomForest", "XGBoost", "SVC", "DecisionTree"]
targets = ['P-LOS', 'POMS3>1', 'Infectious', 'Pulmonary', 'Renal']
data_list = ["CHAR", "CHAR+VAT", "CHAR+FULL"]

# Run optimization for each classifier

for target in targets:
    for data_included in data_list:
        for classifier_name in classifiers:
            result = run_optimization_for_classifier(classifier_name, target, data_included, n_trials=200)
            results.append(result)

# Convert the results dictionary into a DataFrame and save as CSV
results_df = pd.DataFrame(results)

results_df.to_csv(f"metrics.csv", index=False)