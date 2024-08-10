def train(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for features, labels in train_loader:
        predictions = model(features)
        loss = criterion(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)
