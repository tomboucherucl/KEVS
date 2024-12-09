from predictors.kevs_predictions import kevs_predictions
from predictors.thresholding_predictions import thresholding_predictions
from predictors.totalsegmentator_predictions import totalsegmentator_inference

def predict_vat(scans_dir, saros_mask_dir, umamba_predictions_dir, ts_output_dir):
    # Make thresholding predictions
    thresholding_predictions(scans_dir, saros_mask_dir)
    
    # Make KEVS predictions
    kevs_predictions(scans_dir, umamba_predictions_dir)
    
    # Make TotalSegmentator predictions
    totalsegmentator_inference(scans_dir, ts_output_dir)
    