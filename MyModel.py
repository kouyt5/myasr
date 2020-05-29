import torch
import torch.nn.functional as F
import torch.nn as nn
import math


class SeprationConv(nn.Module):
    def __init__(self):
        super(SeprationConv).__init__()

    def forward(self, input):
        pass


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(8, 12), stride=(1, 1),
                      padding=(4, 6)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 8), (1, 1), (2, 4)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.maskcnn = MaskCNN()
        x = math.ceil((float)(64-8+1+4*2))
        x = math.ceil((float)(x-4+1+2*2))
        self.rnn = nn.Sequential(
            nn.LSTM(64*x, 512, num_layers=1,
                    batch_first=True, bidirectional=True),
        )
        self.rnn2 = nn.Sequential(
            nn.LSTM(1024, 256, num_layers=1,
                    batch_first=True, bidirectional=True),
        )
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc = nn.Linear(1024, 29)

    def forward(self, input, percents):
        x = self.cnn(input)  # N*32*F*T
        x = self.maskcnn(x, percents)
        x = x.view(x.size(0), x.size(1)*x.size(2), -
                   1).transpose(1, 2).contiguous()
        x = nn.utils.rnn.pack_padded_sequence(x, enforce_sorted=False,
                                              lengths=torch.mul(x.size(1), percents).int(), batch_first=True)
        x, h = self.rnn(x)
        # x, h = self.rnn2(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)  # N*T*C
        x = x.transpose(1, 2)  # N*C*T
        x = self.bn1(x)
        x = self.fc(x.transpose(1, 2))  # N*T*class
        x = nn.functional.log_softmax(x, dim=-1)
        return x


class MaskCNN(nn.Module):
    def forward(self, x, percents):
        lengths = torch.mul(x.size(3), percents).int()
        mask = torch.BoolTensor(x.size()).fill_(0)
        if x.is_cuda:
            mask = mask.cuda()
        for i, length in enumerate(lengths):
            length = length.item()
            if (mask[i].size(2) - length) > 0:
                mask[i].narrow(
                    2, length, mask[i].size(2) - length).fill_(1)
        x = x.masked_fill(mask, 0)
        return x


if __name__ == "__main__":
    input = torch.rand([8, 1, 64, 128], dtype=torch.float32)
    percents = torch.rand([8], dtype=torch.float32)
    model = MyModel()
    out = model(input, percents)
    print("hh")
