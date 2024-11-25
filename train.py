import torch
from tqdm import tqdm


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=False)
        for batch_idx, (x_temporal, x_topology, targets) in enumerate(train_loader_tqdm):
            x_temporal, x_topology, targets = x_temporal.to(device), x_topology.to(device), targets.to(device)

            # Forward pass
            optimizer.zero_grad()
            output = model(x_temporal, x_topology)
            loss = criterion(output, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

        # Optionally, evaluate on the validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_temporal, x_topology, targets in val_loader:
                x_temporal, x_topology, targets = x_temporal.to(device), x_topology.to(device), targets.to(device)
                output = model(x_temporal, x_topology)
                val_loss += criterion(output, targets).item()

        print(f"Validation Loss: {val_loss / len(val_loader):.4f}")
        model.train()  # Set model back to training mode


def test(model, test_loader, criterion, device):
    model.to(device)
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x_temporal, x_topology, targets in test_loader:
            x_temporal, x_topology, targets = x_temporal.to(device), x_topology.to(device), targets.to(device)

            # Forward pass
            output = model(x_temporal, x_topology)
            test_loss += criterion(output, targets).item()

            # Calculate accuracy
            _, predicted = torch.max(output, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    avg_test_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total

    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    return avg_test_loss, accuracy
