from mccp_lib.dynamic_mc.dynamic_mc import dynamic_mc_predict
from mccp_lib.conformal_prediction.raps import raps_cp
from mccp_lib.conformal_prediction.cqr import cqr_cp

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

'''
MCCP for classification.
'''
def mccp_classification(val_loader, model, device, q_hat, reg_vec, patience=10, min_delta=5e-4, max_mc=1000):
    """
    Apply Monte Carlo Conformal Prediction on the validation set.
    Returns:
      pred_sets: list of prediction sets (class indices) for each validation sample
      label_set: list of true labels for each sample
      error: miscoverage rate (1 - coverage)
    """
    # 1. Get MC dropout predictions for validation set
    val_probs, val_labels = dynamic_mc_predict(model, device, val_loader, max_mc=max_mc, min_delta=min_delta, patience=patience)
    
    # 2. Generate prediction sets for each sample using calibrated q_hat
    pred_sets = raps_cp(val_probs, q_hat, reg_vec)
    
    # 3. Compute coverage and error
    label_set = val_labels.cpu().tolist()
    covered = 0
    for true_label, pred in zip(label_set, pred_sets):
        if true_label in pred:
            covered += 1
    coverage = covered / len(label_set)
    error = 1.0 - coverage
    return pred_sets, label_set, error