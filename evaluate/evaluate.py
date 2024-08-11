import torch


def evaluate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    all_predictions = []
    with torch.no_grad():
        for features, labels in val_loader:
            predictions = model(features)
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            _, predicted = torch.max(predictions.data, 1)
            all_predictions.extend(predicted.cpu().numpy())

    return total_loss / len(val_loader), all_predictions
