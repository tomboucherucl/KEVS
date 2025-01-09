from scripts.make_vat_predictions import predict_vat
from scripts.util.calculate_metrics import calc_metrics
from scripts.util.calculate_metrics_slices import calculate_stat_significance
from scripts.util.outcome_predictions import run_predictions


if __name__ == "__main__":
    # Predict VAT using thresholding, KEVS, and TotalSegmentator
    #predict_vat()
    
    # Calculate metrics for full abdominal cavity:
    calc_metrics()
    
    # Calculate statistical significance using the single slices:
    calculate_stat_significance()
    
    # Make predictions using bayesian hyperparameter optimisation:
    run_predictions()
    