import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import load_and_split_audio
from hcqm import make_specs, compute_hcqm
from dataset import HDF5Dataset, split_dataset, class_to_bpm


class DeepRhythmModel(nn.Module):
    def __init__(self, num_classes=256):
        super(DeepRhythmModel, self).__init__()

        # input shape is (240, 8, 6)
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels=6, out_channels=128, kernel_size=(4, 6), padding='same')
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(4, 6), padding='same')
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4, 6), padding='same')
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(4, 6), padding='same')
        self.bn4 = nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(120, 6))
        self.bn5 = nn.BatchNorm2d(8)

        self.fc1 = nn.Linear(2904, 256)
        self.elu = nn.ELU()

        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, num_classes)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        x = x.reshape(x.size(0), -1)  # Flatten the tensor

        x = self.dropout(self.elu(self.fc1(x)))

        x = self.fc2(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)




def train(data_path, model_name='deeprhythm'):
    dataset = HDF5Dataset(data_path)
    train_dataset, test_dataset, validate_dataset = split_dataset(dataset)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size=256, shuffle=False)

    # Move the model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)

    # Early stopping setup
    early_stopping_patience = 5
    early_stopping_counter = 0
    best_validate_loss = float('inf')

    # Training loop
    num_epochs = 20
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

        # Check for early stopping
        if average_validate_loss < best_validate_loss:
            best_validate_loss = average_validate_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    model_path = f'{model_name}.pth'
    torch.save(model.state_dict(), model_path)

def load_model(path, device=None):
    model = DeepRhythmModel(256)
    # Load the weights
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device)
    model.eval()
    return model


def predict_global_bpm(input_path, model_path='deeprhythm0.1.pth', model=None, specs=None):
    if model is None:
        model = load_model(model_path)
    clips = load_and_split_audio(input_path, sr=22050)
    model_device = next(model.parameters()).device
    if specs is None:
        stft, band, cqt = make_specs(device=model_device)
    else:
        stft, band, cqt = specs
    input_batch = compute_hcqm(clips.to(device=model_device), stft, band, cqt).permute(0,3,1,2)
    model.eval()
    with torch.no_grad():
        # Ensure the batch is on the same device as the model
        input_batch = input_batch.to(device=model_device)
        outputs = model(input_batch)

        # Apply softmax to convert logits to probabilities
        probabilities = torch.softmax(outputs, dim=1)

        # Compute the average probability across the batch for each class
        mean_probabilities = probabilities.mean(dim=0)

        # Find the class with the maximum average probability
        _, predicted_class = torch.max(mean_probabilities, 0)
        predicted_global_bpm = class_to_bpm(predicted_class.item())

    return predicted_global_bpm