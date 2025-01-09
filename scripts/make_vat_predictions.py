from .util.kevs_predictions import KEVS_prediction
from .util.thresholding_predictions import thresholding_predictions
from .util.totalsegmentator_predictions import totalsegmentator_inference
import os

def predict_vat():
    scans_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "testing", "20_scans")
    # Make thresholding predictions
    thresholding_predictions(scans_dir)
    
    # Make KEVS predictions
    KEVS_prediction()
    
    # Make TotalSegmentator predictions
    totalsegmentator_inference(scans_dir)
    
if __name__ == "__main__":
    predict_vat()