import torch
from tqdm import tqdm


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=False)
        for batch_idx, (x_temporal, x_topology, targets) in enumerate(train_loader_tqdm):
            x_temporal, targets = x_temporal.to(device), targets.to(device)

            # Forward pass
            optimizer.zero_grad()
            output = model(x_temporal, None)
            loss = criterion(output, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

        # Optionally, evaluate on the validation set
        with model.evaluation_mode():  # Use the context manager to set model to eval mode
            val_loss = 0.0
            with torch.no_grad():
                for x_temporal, x_topology, targets in val_loader:
                    x_temporal, targets = x_temporal.to(device), targets.to(device)
                    output = model(x_temporal, None)
                    val_loss += criterion(output, targets).item()

            print(f"Validation Loss: {val_loss / len(val_loader):.4f}")


def test(model, test_loader, criterion, device):
    """
    Evaluate the model on the test set.

    Parameters:
    - model: The model to evaluate.
    - test_loader: DataLoader for the test set.
    - criterion: Loss function.
    - device: Device (CPU or GPU).

    Returns:
    - Average test loss, accuracy, recall, precision, F1 score, and AUC-ROC.
    """
    model.to(device)
    test_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []
    all_probabilities = []

    with model.evaluation_mode():  # Temporarily set model to eval mode
        with torch.no_grad():
            for x_temporal, x_topology, targets in test_loader:
                x_temporal, targets = x_temporal.to(device), targets.to(device)

                output = model(x_temporal, None)
                test_loss += criterion(output, targets).item()

                _, predicted = torch.max(output, 1)
                probabilities = torch.softmax(output, dim=1)[:, 1]

                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

                correct += (predicted == targets).sum().item()
                total += targets.size(0)

    avg_test_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    recall = recall_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)
    auc_roc = roc_auc_score(all_targets, all_probabilities)

    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Recall (Sensitivity): {recall:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"AUC-ROC: {auc_roc:.2f}")

    return avg_test_loss, accuracy, recall, precision, f1, auc_roc