import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from deeprhythm.dataset.clip_dataset import ClipDataset
from deeprhythm.model.frame_cnn import DeepRhythmModel
from deeprhythm.dataset.song_dataset import SongDataset,song_collate
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_cnn(data_path, model_name='deeprhythm', start_weights=None, batch_size=256, early_stopping_patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepRhythmModel()
    if start_weights is not None:
        model.load_state_dict(torch.load(start_weights, map_location=torch.device(device)))
    data_path = '/media/bleu/bulkdata2/deeprhythmdata/hcqm-split.hdf5'
    train_dataset = ClipDataset(data_path, 'train')
    test_dataset = ClipDataset(data_path, 'test')
    validate_dataset = ClipDataset(data_path, 'val')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001,  betas=(0.9, 0.999), eps=1e-8)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)
    early_stopping_counter = 0
    best_validate_loss = float('inf')
    # Training loop
    num_epochs = 40
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation loop
        model.eval()
        validate_loss = 0.0
        with torch.no_grad():
            for inputs, labels in validate_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                validate_loss += loss.item()

        average_train_loss = running_loss / len(train_loader)
        average_validate_loss = validate_loss / len(validate_loader)
        print(f"Epoch {epoch+1}, Train Loss: {average_train_loss:.4f}, Validate Loss: {average_validate_loss:.4f}")
        scheduler.step(average_validate_loss)

        # Check for early stopping
        if average_validate_loss < best_validate_loss:
            best_validate_loss = average_validate_loss
            early_stopping_counter = 0
            # save the best version of the model
            model_path = f'{model_name}-best.pth'
            torch.save(model.state_dict(), model_path)
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break
    test_loss = 0.0
    correct1 = 0
    correct2 = 0
    total_predictions = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_predictions += labels.size(0)
            for prediction, label in zip(predicted, labels):
                tolerance = 0.04 * label.item()
                if abs(prediction.item() - label.item()) <= tolerance:
                    correct1 += 1
                for multiple in range(1, int(label.item() / (label.item() * tolerance)) + 1):
                    if abs(prediction.item() - (label.item() * multiple)) <= tolerance:
                        correct2 += 1
                        break

    average_test_loss = test_loss / len(test_loader)
    accuracy1 = correct1 / total_predictions
    accuracy2 = correct1 / total_predictions
    print(f"Test Loss: {average_test_loss:.4f}, Accuracy1: {accuracy1:.4f}, Accuracy2: {accuracy2:.4f}")



def train_cnn_cont(data_path, model_name='deeprhythm', start_weights=None, batch_size=256, early_stopping_patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepRhythmModel()
    if start_weights is not None:
        model.load_state_dict(torch.load(start_weights, map_location=torch.device(device)))
    data_path = '/media/bleu/bulkdata2/deeprhythmdata/hcqm-split.hdf5'
    train_dataset = ClipDataset(data_path, 'train', use_float=True)
    test_dataset = ClipDataset(data_path, 'test', use_float=True)
    validate_dataset = ClipDataset(data_path, 'val', use_float=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = model.to(device)

    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001,  betas=(0.9, 0.999), eps=1e-8)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)
    early_stopping_counter = 0
    best_validate_loss = float('inf')
    # Training loop
    num_epochs = 40
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # Calculate expected BPM for each output
            probabilities = torch.softmax(outputs, dim=1)
            expected_bpm = torch.sum(probabilities * torch.arange(256).float().to(device), dim=1)
            loss = criterion(expected_bpm, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation loop
        model.eval()
        validate_loss = 0.0
        with torch.no_grad():
            for inputs, labels in validate_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                expected_bpm = torch.sum(probabilities * torch.arange(256).float().to(device), dim=1)
                loss = criterion(expected_bpm, labels)
                validate_loss += loss.item()

        average_train_loss = running_loss / len(train_loader)
        average_validate_loss = validate_loss / len(validate_loader)
        print(f"Epoch {epoch+1}, Train Loss: {average_train_loss:.4f}, Validate Loss: {average_validate_loss:.4f}")
        scheduler.step(average_validate_loss)

        # Check for early stopping
        if average_validate_loss < best_validate_loss:
            best_validate_loss = average_validate_loss
            early_stopping_counter = 0
            # save the best version of the model
            model_path = f'{model_name}-best.pth'
            torch.save(model.state_dict(), model_path)
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break
    test_loss = 0.0
    correct1 = 0
    correct2 = 0
    total_predictions = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            expected_bpm = torch.sum(probabilities * torch.arange(256).float().to(device), dim=1)
            loss = criterion(expected_bpm, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_predictions += labels.size(0)
            for prediction, label in zip(predicted, labels):
                tolerance = 0.04 * label.item()
                if abs(prediction.item() - label.item()) <= tolerance:
                    correct1 += 1
                for multiple in range(1, int(label.item() / (label.item() * tolerance)) + 1):
                    if abs(prediction.item() - (label.item() * multiple)) <= tolerance:
                        correct2 += 1
                        break

    average_test_loss = test_loss / len(test_loader)
    accuracy1 = correct1 / total_predictions
    accuracy2 = correct1 / total_predictions
    print(f"Test Loss: {average_test_loss:.4f}, Accuracy1: {accuracy1:.4f}, Accuracy2: {accuracy2:.4f}")
