import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, alphabet_len, hidden_size=256):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(512, 512, 2, 1), nn.ReLU(True)
        )

        self.rnn = nn.LSTM(input_size=512, hidden_size=hidden_size,
                           num_layers=2, bidirectional=True, batch_first=True)

        self.fc = nn.Linear(hidden_size*2, alphabet_len + 1)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1, "CNN output height must be 1"
        conv = conv.squeeze(2).permute(0, 2, 1)
        y, _ = self.rnn(conv)
        out = self.fc(y)
        return out 