import argparse
import logging
from logging import handlers

from sklearn.metrics import f1_score

import torch
import torch.nn as nn


class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }
    def __init__(self, 
                 filename, 
                 level='info',
                 when='D',
                 backCount=3,
                 fmt='%(asctime)s - %(pathname)s - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        th = handlers.TimedRotatingFileHandler(filename=filename,
                                               when=when,
                                               backupCount=backCount,
                                               encoding='utf-8')
        th.setFormatter(format_str)
        self.logger.addHandler(sh) 
        self.logger.addHandler(th)

# Evaluate the model on the given dataloader and compute accuracy, loss, and F1 score.
def eval_model_dataloder(model, dataloader, device='cpu'):
    model.to(device)
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)

            # Store labels and predictions for metric computation
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Compute average loss
    avg_loss = running_loss / len(dataloader.dataset)

    # Compute accuracy
    correct_predictions = sum(p == t for p, t in zip(all_preds, all_labels))
    accuracy = correct_predictions / len(all_labels)

    # Compute F1 score
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return accuracy, avg_loss, f1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=10, help='The number of the MCCP iterations (default: 10).')
    parser.add_argument('--patience', type=int, default=10, help='Patience for adaptive MC (default: 10).')
    parser.add_argument('--n-cal', type=int, default=2500, help='The number of samples to form the calibration set(default: 2500).')
    parser.add_argument('--min-delta', type=float, default=5e-4, help='Delta (i.e., the minimum varience difference) for adaptive MC (default: 5e-4).')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training.')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--logging', action="store_true", help="Whether to log process")


    args = parser.parse_args()
    # print(args)
    
    return args