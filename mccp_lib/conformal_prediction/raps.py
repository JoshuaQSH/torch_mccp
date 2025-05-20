import torch

'''
RAPS calibration.
'''
def raps_calibration_old(cal_softmax, y_cal, reg_vec, alpha=0.05):
    N, C = cal_softmax.shape
    device = cal_softmax.device
    pi = torch.argsort(cal_softmax, dim=1, descending=True)
    srt = torch.gather(cal_softmax, 1, pi)
    srt_reg = srt + reg_vec.to(device)

    label_pos = (pi == y_cal.unsqueeze(1)).nonzero(as_tuple=False)[:, 1]
    cum = torch.cumsum(srt_reg, dim=1)
    base_score = cum[torch.arange(N), label_pos]
    noise = torch.rand(N, device=device) * srt_reg[torch.arange(N), label_pos]
    cal_scores = base_score - noise

    q_level = torch.ceil((N + 1) * (1 - alpha)) / N
    q_hat = torch.quantile(cal_scores, q_level.item(), interpolation='higher')
    return q_hat

def raps_calibration(probs: torch.Tensor, labels: torch.Tensor, reg_vec: torch.Tensor, alpha=0.05):
    # Number of calibration samples and classes
    n_cal, n_classes = probs.size(0), probs.size(1)
    # Get the probability assigned to the true label for each sample
    true_probs = probs[torch.arange(n_cal), labels]
    # Determine the rank of the true label in descending order (1 = highest rank)
    # rank_i = 1 + number of classes with probability > true_prob
    ranks = (probs > true_probs.unsqueeze(1)).sum(dim=1) + 1
    # Calculate cumulative probability up to and including the true label
    # (sum of probabilities of all classes with prob >= true_prob)
    cum_probs = true_probs + (probs * (probs > true_probs.unsqueeze(1)).float()).sum(dim=1)
    # Compute RAPS score = cum_prob + penalty for classes beyond k_reg
    # Penalty = lam_reg * max(ranks - k_reg, 0)
    penalty = reg_vec[(ranks - 1).clamp(min=0)] * torch.clamp(ranks - (reg_vec == 0).sum(), min=0)
    # (reg_vec == 0).sum() gives k_reg (count of zero-penalty classes).
    scores = cum_probs + penalty
    # Determine the (1 - alpha) quantile of scores (conformal cutoff)
    sorted_scores, _ = torch.sort(scores)
    # Index for quantile: ceil((n_cal + 1) * (1 - alpha)) - 1 (0-indexed)
    quantile_index = int(torch.ceil((n_cal + 1) * torch.tensor(1 - alpha)).item() - 1)
    quantile_index = max(0, min(n_cal - 1, quantile_index))  # clamp within [0, n_cal-1]
    q_hat = sorted_scores[quantile_index]
    return q_hat

'''
RAPS prediction.
'''
def raps_cp_old(q_hat, softmax, reg_vec, y_test=None, disallow_zero_sets=False, rand=False):
    N, C = softmax.shape
    device = softmax.device

    pi = torch.argsort(softmax, dim=1, descending=True)
    srt = torch.gather(softmax, 1, pi)
    srt_reg = srt + reg_vec.to(device)
    cum = torch.cumsum(srt_reg, dim=1)

    if rand:
        sampled = torch.rand(N, device=device).unsqueeze(1) * srt_reg
        mask = cum - sampled <= q_hat
    else:
        mask = (cum - srt_reg) <= q_hat

    if disallow_zero_sets:
        mask[:, 0] = True

    inv_pi = torch.argsort(pi, dim=1)
    pred_mask = torch.gather(mask, 1, inv_pi)

    if y_test is not None:
        correct = pred_mask[torch.arange(N), y_test]
        acc = correct.float().mean().item()
        return pred_mask, acc
    return pred_mask, None


def raps_cp(probs: torch.Tensor, q_hat: torch.Tensor, reg_vec: torch.Tensor):
    N, C = probs.size()
    # Sort class probabilities in descending order for each sample
    sorted_probs, sorted_idx = torch.sort(probs, dim=1, descending=True)
    # Cumulative sum of probabilities along sorted classes
    cum_probs = torch.cumsum(sorted_probs, dim=1)
    # Pre-compute cumulative penalty: reg_cum[k] = sum of reg_vec for first k classes
    reg_cum = torch.cumsum(reg_vec, dim=0)
    # Determine for each sample the smallest k such that cum_prob + penalty >= q_hat
    # Compute matrix of cum_prob + penalty for each possible k (for vectorization)
    # reg_cum is 1-D of length C; we broadcast it to shape (N, C) for addition
    total_score = cum_probs + reg_cum.unsqueeze(0)
    # Create a boolean mask where condition is satisfied
    meet_threshold = total_score >= q_hat.to(probs.device)
    # Find the first index where condition is True for each sample
    # (Since total_score is non-decreasing in k, argmax of True values gives first True)
    first_true_idx = meet_threshold.float().argmax(dim=1)
    # Build the prediction sets
    pred_sets = []
    for i in range(N):
        k = int(first_true_idx[i].item())  # number of classes to include (index 0 -> 1 class)
        if not meet_threshold[i, k]:  
            # If threshold never met (shouldn't happen if q_hat is calibrated within reachable range),
            # include all classes
            k = C - 1
        # Take top-(k+1) classes as prediction set for sample i
        pred_labels = sorted_idx[i, :k+1].tolist()
        pred_sets.append(pred_labels)
    return pred_sets