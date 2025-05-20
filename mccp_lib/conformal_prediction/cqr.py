import numpy as np
import tqdm
from sklearn.metrics import mean_absolute_error as mae


'''
CQR calibration.
'''
def cqr_calibration(cal_upper, cal_lower, cal_labels, n=10, alpha=0.1):
    cal_scores = np.maximum(cal_labels-cal_upper, cal_lower-cal_labels)

    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_hat = np.quantile(cal_scores, q_level, method="higher")

    return q_hat


'''
CQR prediction.
'''
def cqr_cp(val_upper, val_lower, q_hat, y_test=None):
    prediction_sets = [val_lower - q_hat, val_upper + q_hat]

    if y_test is not None:
        # coverage calculation
        prediction_sets_uncalibrated = [val_lower, val_upper]

        empirical_coverage_uncalibrated = ((y_test >= prediction_sets_uncalibrated[0]) & \
            (y_test <= prediction_sets_uncalibrated[1])).mean()
        # uncal_cov.append(empirical_coverage_uncalibrated)
        empirical_coverage = ((y_test >= prediction_sets[0]) & (y_test <= \
            prediction_sets[1])).mean()
        # cal_cov.append(empirical_coverage)

    upper_preds = []
    lower_preds = []
    # predictions
    for idx in tqdm.tqdm(range(len(val_lower))):
        upper_pred = prediction_sets[1][idx]
        lower_pred = prediction_sets[0][idx]

        upper_preds.append(upper_pred)
        lower_preds.append(lower_pred)

    if y_test is not None:
        # mae calculation
        upper_mae = mae(y_test, upper_preds)
        lower_mae = mae(y_test, lower_preds)
        full_mae = upper_mae + lower_mae
        # test_maes.append(full_mae)
    if y_test is not None:
        return prediction_sets, upper_preds, lower_preds, full_mae, \
            empirical_coverage_uncalibrated, empirical_coverage
    return prediction_sets, upper_preds, lower_preds, None, None, None