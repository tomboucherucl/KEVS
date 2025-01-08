from predictors.kevs_predictions import KEVS_prediction
from predictors.thresholding_predictions import thresholding_predictions
from predictors.totalsegmentator_predictions import totalsegmentator_inference
import os

def predict_vat(scans_dir, abdominal_mask_dir, umamba_predictions_dir):
    # Make thresholding predictions
    #thresholding_predictions(scans_dir, abdominal_mask_dir)
    
    # Make KEVS predictions
    KEVS_prediction()
    
    # Make TotalSegmentator predictions
    #totalsegmentator_inference(scans_dir)
    
if __name__ == '__main__':
    scans_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "filtered_scans_relabelled")
    abdominal_cavity_masks_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "abdominal_cavity_mask")
    umamba_predictions_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "umamba_predictions")

    predict_vat(scans_path, abdominal_cavity_masks_path, umamba_predictions_path)