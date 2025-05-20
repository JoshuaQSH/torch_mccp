import torch
from mccp_lib.dynamic_mc.dynamic_mc import dynamic_mc_predict
from mccp_lib.conformal_prediction.raps import raps_calibration


'''
Conformal prediction calibration.
'''
def cp_calibration_old(cal_loader, model, patience, min_delta, max_mc,
                         k_reg=5, lam_reg=0.01, alpha=0.05, device='cuda'):

    y_cal = torch.cat([y for _, y in cal_loader], dim=0).to(device)
    mc_preds = dynamic_mc_predict(cal_loader, model, patience, min_delta, max_mc, device=device)
    reg_vec = torch.cat([
        torch.zeros(k_reg, device=device),
        torch.full((mc_preds.shape[1] - k_reg,), lam_reg, device=device)
    ]).unsqueeze(0)
    q_hat = raps_calibration(mc_preds, y_cal, reg_vec, alpha=alpha)
    return q_hat, reg_vec


"""
New conformal prediction calibration.
"""
def cp_calibration(cal_loader, model, device, patience=10,  min_delta=5e-4, max_mc=1000, k_reg=5, lam_reg=0.01, alpha=0.05):
    """
    Calibrate the Monte Carlo Conformal Predictor using a calibration dataset.
    Returns:
      q_hat: calibrated RAPS threshold
      reg_vec: regularization vector (penalty per class position)
    """
    # 1. Get MC dropout predictions for calibration set
    cal_probs, cal_labels = dynamic_mc_predict(model, device, cal_loader, max_mc=max_mc, patience=patience, min_delta=min_delta)
    
    # 2. Construct regularization vector for RAPS
    num_classes = cal_probs.size(1)
    reg_vec = torch.zeros(num_classes, device=device)
    if k_reg < num_classes:
        reg_vec[k_reg:] = lam_reg
        
    # 3. Calibrate q_hat using RAPS scores on calibration data
    q_hat = raps_calibration(cal_probs, cal_labels, reg_vec, alpha)
    return q_hat, reg_vec