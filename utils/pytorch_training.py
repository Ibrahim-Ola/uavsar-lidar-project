import numpy as np

import torch
import torch.optim as optim
from torch.nn.functional import l1_loss, mse_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, val_loader, epochs, criterion, optimizer, device, metric='mae'):
    model.to(device)
    history = {'train_loss': [], 'val_loss': [], 'train_metric': [], 'val_metric': []}

    # 1. Instantiate the ReduceLROnPlateau scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1, verbose=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_metric = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            
            if metric == 'mae':
                metric_val = l1_loss(outputs, labels)
            elif metric == 'rmse':
                metric_val = torch.sqrt(mse_loss(outputs, labels))
            else:
                raise ValueError("Invalid metric. Choose either 'mae' or 'rmse'.")
            
            train_metric += metric_val.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_metric = train_metric / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        history['train_metric'].append(avg_train_metric)

        model.eval()
        val_loss = 0
        val_metric = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)

                if metric == 'mae':
                    metric_val = l1_loss(outputs, labels)
                elif metric == 'rmse':
                    metric_val = torch.sqrt(mse_loss(outputs, labels))
                else:
                    raise ValueError("Invalid metric. Choose either 'mae' or 'rmse'.")
                
                val_metric += metric_val.item()
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_metric = val_metric / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        history['val_metric'].append(avg_val_metric)

        # 2. Step the scheduler with validation loss for ReduceLROnPlateau
        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Train {metric.upper()}: {avg_train_metric:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation {metric.upper()}: {avg_val_metric:.4f}")

    return history



def predict(model, test_loader, device):
    model.eval()
    model.to(device)
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = outputs.squeeze().cpu().numpy()
            all_predictions.extend(outputs)
            all_labels.extend(labels.squeeze().cpu().numpy()) 

    return {
        'predictions': np.array(all_predictions),
        'labels': np.array(all_labels)
    }
