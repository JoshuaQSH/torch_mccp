from mccp_lib.datasets.load import load_cifar10
from mccp_lib.models.cnn import CNN_MC
from mccp_lib.utils.proc_data import split_test_set
from mccp_lib.utils.utils import parse_args, eval_model_dataloder, Logger
from mccp_lib.mccp.calibration import cp_calibration
from mccp_lib.mccp.prediction import mccp_classification


import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from tqdm import tqdm

def train_cnn_model(model, train_loader, epochs=10, lr=0.01, device='cuda:0', verbose=1):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.to(device)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        # Wrap train_loader with tqdm for a progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            # Update tqdm progress bar with loss and accuracy
            progress_bar.set_postfix(loss=loss.item(), accuracy=correct_predictions / total_predictions if total_predictions > 0 else 0)
        
        avg_loss = running_loss / len(train_loader.dataset)
        accuracy = correct_predictions / total_predictions
        
        if verbose:
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
def run(args, model, device, cal_loader, val_loader):
    
    # Calibration phase
    q_hat, reg_vec = cp_calibration(cal_loader, 
                                    model,
                                    device=device,
                                    patience=args.patience, 
                                    min_delta=args.min_delta,
                                    max_mc=1000, 
                                    k_reg=5, 
                                    lam_reg=0.01, 
                                    alpha=0.05)
    
    # Valiudation phase and generate prediction sets with calibrated q_hat
    pred_set, label_set, error = mccp_classification(val_loader, 
                                                     model,
                                                     device,
                                                     q_hat, 
                                                     reg_vec,
                                                     patience=args.patience,
                                                     min_delta=args.min_delta,
                                                     max_mc=1000)
    return pred_set, label_set, error    

if __name__ == "__main__":
    args = parse_args()
    
    train_loader, test_loader = load_cifar10(batch_size=args.batch_size, shuffle=True, num_workers=2)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else "cpu")

    # device = torch.device("cpu")
    model = CNN_MC(input_shape=(3, 32, 32), output_dims=10, montecarlo=True)

    train_cnn_model(model, train_loader=train_loader, epochs=args.epochs, lr=0.01, device=device)
    
    errors = []
    
    for i in range(args.iterations):
        # split test set into calibration and validation set
        cal_loader, val_loader = split_test_set(test_loader.dataset, n_cal=args.n_cal, batch_size=args.batch_size, shuffle=False)
        
        pred_set, label_set, error = run(args, model, device, cal_loader, val_loader)
        errors.append(error)
        
    print("Test error: {:2f}, +/-{:2f}".format(np.mean(errors), np.std(errors)))
