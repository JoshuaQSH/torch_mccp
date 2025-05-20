import torch
from tqdm import tqdm

def dynamic_mc_predict(model, device, data_loader, max_mc=1000, min_delta=5e-4, patience=10):
    """
    Perform adaptive Monte Carlo Dropout inference on all data in the loader.
    Returns the mean softmax predictions (i.e., [class1_mean, class2_mean, ..., classn_mean]) and true labels.
    Returns:
      - all_preds: Tensor of shape (N_samples, num_classes) with mean softmax predictions.
      - all_labels: Tensor of shape (N_samples,) with the true labels.
    """
    model = model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    
    # Disable gradient computation for inference
    with torch.no_grad():
        for images, labels in tqdm(data_loader):
        # for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            # First forward pass
            outputs = torch.softmax(model(images), dim=1)  # softmax probabilities
            batch_mean = outputs.clone()               # running mean of predictions
            batch_M2 = torch.zeros_like(outputs)       # running sum of squared diff (for variance)
            
            # Initialize variance change tracking
            prev_variance = torch.zeros_like(outputs)
            stable_count = torch.zeros(images.size(0), device=device, dtype=torch.long)
            
            # Additional MC dropout passes up to max_passes
            for t in range(2, max_mc + 1):
                # Perform another forward pass with dropout
                outputs = torch.softmax(model(images), dim=1)
                
                # Update running mean and M2 (variance) for each sample and class (Welford's method)
                delta = outputs - batch_mean
                batch_mean = batch_mean + delta / t
                batch_M2 = batch_M2 + delta * (outputs - batch_mean)
                
                # Compute the new variance
                variance = batch_M2 / t
                
                # Check change in variance from previous iteration
                var_diff = (variance - prev_variance).abs()
                prev_variance = variance
                
                # Update stability count: increment if all class variances changed <= min_delta, else reset
                mask_converged = (var_diff <= min_delta).all(dim=1)
                stable_count[mask_converged] += 1
                stable_count[~mask_converged] = 0
                
                # Stop if all samples in this batch have been stable for 'patience' passes
                if (stable_count >= patience).all():
                    break
            
            # Collect the final mean predictions and labels for this batch
            all_preds.append(batch_mean)  
            all_labels.append(labels)
            
    # Concatenate batch results
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return all_preds, all_labels

def dynamic_mc_predict_batch(model, device, data_loader, num_classes=10, max_passes=1000, delta=5e-4, patience=10):
    """
    Runs adaptive Monte Carlo dropout prediction.
    Returns:
      - Tensor of shape (N, num_classes) with mean probabilities.

    """
    model.eval()
    model = model.to(device)
        
    # mean probabilities for each batch
    all_means = []
    for batch in data_loader:
        # Support data_loader yielding (inputs, labels) or just inputs
        if isinstance(batch, (list, tuple)) and len(batch) >= 1:
            inputs = batch[0]
        else:
            inputs = batch
        inputs = inputs.to(device)
        batch_size = inputs.size(0)
        
        # Accumulate predictions for this batch
        preds_sum = torch.zeros(batch_size, num_classes, device=device)  # sum of probabilities
        preds_sum_sq = torch.zeros(batch_size, num_classes, device=device)  # sum of squared probabilities (for variance)
        
        # Track variance convergence
        prev_variance = torch.zeros(batch_size, num_classes, device=device)
        # Patience counter per sample
        consec_passes_below_delta = torch.zeros(batch_size, dtype=torch.int32, device=device)
        # Active mask: which samples are still running MC passes
        active = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        # Main MC pass
        for t in range(1, max_passes+1):
            # Model Forward pass
            with torch.no_grad():
                out = model(inputs[active])  # shape: (active_count, num_classes)
                probs = torch.softmax(out, dim=1)
            
            # Expand `probs` back to full batch shape by placing into correct indices
            if active.all():
                # All samples active (common in initial passes)
                current_probs = probs  # shape: (batch_size, num_classes)
            else:
                # Some samples finished early, we insert dummy zeros for them
                current_probs = torch.zeros(batch_size, num_classes, device=device)
                current_probs[active] = probs
            
            # Accumulate for mean/variance
            preds_sum += current_probs
            preds_sum_sq += current_probs ** 2

            # Update variance for active samples
            mean = preds_sum / t
            var = preds_sum_sq / t - mean**2
            # Compute max variance change from previous pass for each sample
            var_change = (var - prev_variance).abs().max(dim=1).values  # max change across classes
            prev_variance = var.clone()
            
            # Update patience counters for active samples
            # If variance change below delta, increment counter; else reset counter
            below_thresh = (var_change < delta)
            # Only consider active samples for patience (others are done, but still in tensor form)
            consec_passes_below_delta[active] = torch.where(below_thresh[active],
                                                            consec_passes_below_delta[active] + 1,
                                                            torch.zeros_like(consec_passes_below_delta[active]))
            
            # Determine which active samples have met patience criterion
            done_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            done_mask[active] = consec_passes_below_delta[active] >= patience
            
            # Mark those samples as no longer active
            if done_mask.any():
                active[done_mask] = False
            # If all samples in batch are done, break early
            if not active.any():
                break

        # Compute final mean probabilities for this batch
        # t is either max_passes or the pass we broke out on for each (but we didn't track per sample break count here)
        mean_probs = preds_sum / t  
        all_means.append(mean_probs.cpu())

    # Combine results from all batches
    predictions = torch.cat(all_means, dim=0)  # shape: (N, C)
    return predictions

