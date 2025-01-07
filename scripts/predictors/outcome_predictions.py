import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import logging
import matplotlib.pyplot as plt

logging.basicConfig(filename='outcome_predictions.log', level=logging.INFO, format='%(asctime)s %(message)s')

def read_data(target, data_included):
    if target == 'P-LOS':
        column_choice = -3
    elif target == 'POMS3>1':
        column_choice = -2
    elif target == 'Infectious':
        column_choice = -1
    
    data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "csv", f"predictions.csv" ))
    y_data = data[data.columns[column_choice]]  # The last column is the target variable
    x_data = data.drop(labels=[data.columns[-3:-1]], axis=1)  # Drop the last three columns (targets)
    
    if data_included == 'CHAR':
        continue
    elif data_included == 'CHAR+VAT':
        continue
    elif data_included == 'CHAR+FULL':
        continue
    return np.array(x_data), np.array(y_data)

# Define the objective function for hyperparameter optimization
def objective(trial, classifier_type, target, data_included):
    # Load the dataset using the custom function
    X, y = read_data(target, data_included)

    # Suggest whether to apply PCA or not
    pca = trial.suggest_categorical("pca", [True, False])
    
    # Standardize the dataset
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Apply PCA if selected
    if pca:
        pca_model = PCA(n_components=None)
        X = pca_model.fit_transform(X)

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
        classifier = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, eval_metric='logloss', device='cuda:1')

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
    mean_recall, std_recall = np.mean(recall_scores), np.std(recall_scores)
    mean_precision, std_precision = np.mean(precision_scores), np.std(precision_scores)
    mean_accuracy, std_accuracy = np.mean(accuracy_scores), np.std(accuracy_scores)

    # Store additional metrics in the trial
    trial.set_user_attr("roc_auc_mean", mean_auc)
    trial.set_user_attr("roc_auc_std", std_auc)
    trial.set_user_attr("f1_mean", mean_f1)
    trial.set_user_attr("f1_std", std_f1)
    trial.set_user_attr("recall_mean", mean_recall)
    trial.set_user_attr("recall_std", std_recall)
    trial.set_user_attr("precision_mean", mean_precision)
    trial.set_user_attr("precision_std", std_precision)
    trial.set_user_attr("accuracy_mean", mean_accuracy)
    trial.set_user_attr("accuracy_std", std_accuracy)

    # Optimizing for the product of mean ROC AUC and mean F1-score
    return mean_auc

# Create an Optuna study to maximize the product of ROC AUC and F1 score
def run_optimization_for_classifier(classifier_name, target, data_included, n_trials=500):
    logging.info(f"Optimising {classifier_name}...")
    study = optuna.create_study(direction="maximize", study_name=f"{classifier_name} optimisation for {target} with {data_included} data.")
    study.optimize(lambda trial: objective(trial, classifier_name, target, data_included), n_trials=n_trials, n_jobs=8)
    
    # Get the best trial for this classifier
    best_trial = study.best_trial
    
    logging.info(f"Classifier: {classifier_name}")
    logging.info(f"Target: {target}")
    logging.info(f"Data included: {data_included}")
    logging.info(f"Best ROC AUC (mean ± std): {best_trial.user_attrs['roc_auc_mean']:.4f} ± {best_trial.user_attrs['roc_auc_std']:.4f}")
    logging.info(f"Best F1 Score (mean ± std): {best_trial.user_attrs['f1_mean']:.4f} ± {best_trial.user_attrs['f1_std']:.4f}")
    logging.info(f"Best Recall (mean ± std): {best_trial.user_attrs['recall_mean']:.4f} ± {best_trial.user_attrs['recall_std']:.4f}")
    logging.info(f"Best Precision (mean ± std): {best_trial.user_attrs['precision_mean']:.4f} ± {best_trial.user_attrs['precision_std']:.4f}")
    logging.info(f"Best Accuracy (mean ± std): {best_trial.user_attrs['accuracy_mean']:.4f} ± {best_trial.user_attrs['accuracy_std']:.4f}")

    # Return the metrics of the best trial
    return {
        "Classifier": classifier_name,
        "Target": target,
        "Data included": data_included,
        "Best ROC AUC (mean ± std)": f"{best_trial.user_attrs['roc_auc_mean']:.4f} ± {best_trial.user_attrs['roc_auc_std']:.4f}",
        "Best F1 Score (mean ± std)": f"{best_trial.user_attrs['f1_mean']:.4f} ± {best_trial.user_attrs['f1_std']:.4f}",
        "Best Recall (mean ± std)": f"{best_trial.user_attrs['recall_mean']:.4f} ± {best_trial.user_attrs['recall_std']:.4f}",
        "Best Precision (mean ± std)": f"{best_trial.user_attrs['precision_mean']:.4f} ± {best_trial.user_attrs['precision_std']:.4f}",
        "Best Accuracy (mean ± std)": f"{best_trial.user_attrs['accuracy_mean']:.4f} ± {best_trial.user_attrs['accuracy_std']:.4f}"
    }

# Dictionary to store the results for all classifiers
results = []

# List of classifiers to optimize
classifiers = ["LogisticRegression", "RandomForest", "XGBoost", "SVC", "DecisionTree"]
targets = ["P-LOS", "POMS3>1", "Infectious"]
data_list = ["CHAR", "CHAR+VAT", "CHAR+FULL"]

# Run optimization for each classifier

for target in targets:
    for data_included in data_list:
        for classifier_name in classifiers:
            result = run_optimization_for_classifier(classifier_name, data_included, target, n_trials=500)
            results.append(result)

    # Convert the results dictionary into a DataFrame and save as CSV
    results_df = pd.DataFrame(results)

    results_df.to_csv(f"metrics_{target}.csv", index=False)

    # Output the best results
    print(f"Best results saved to metrics_{target}.csv")
    print(results_df)
