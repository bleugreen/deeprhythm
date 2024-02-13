import torch.nn as nn
import torch.nn.functional as F

class DeepRhythmModel(nn.Module):
    def __init__(self, num_classes=256):
        super(DeepRhythmModel, self).__init__()
        # input shape is (6, 240, 8)
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
        self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = x.reshape(x.size(0), -1)
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