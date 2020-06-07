import torch
import torch.nn.functional as F
import torch.nn as nn
import math


class SeprationConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=(1,33),last=False):
        super(SeprationConv,self).__init__()
        self.last = last
        self.depthwise_conv = nn.Conv2d(in_ch, in_ch, kernel_size=k, stride=(1, 1),
                      padding=(k[0]//2, k[1]//2), groups=in_ch)
        self.pointwise_conv = nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1), stride=(1, 1))
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        # self.maskcnn = MaskCNN()
    def forward(self, input):
        x = self.depthwise_conv(input)
        x = self.channel_shuffle(x, groups=4)
        x = self.pointwise_conv(x)
        # x = self.maskcnn(x)
        x = self.bn(x)
        if not self.last:
            x= self.relu(x)
        return x

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, f, time = x.data.size()
        channels_per_group = num_channels // groups
        if not channels_per_group * groups == num_channels:
            raise "group数和通道数不匹配，请保证group能够被num_channels整除"
        # reshape
        x = x.view(batchsize, groups,
               channels_per_group, f, time)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batchsize, -1, f, time)
        return x

class QuartNetBlock(nn.Module):
    def __init__(self, repeat=3,in_ch=1,out_ch=32,k=(1,33)):
        super(QuartNetBlock, self).__init__()
        seq = []
        for i in range(0,repeat-1):
            sep = SeprationConv(in_ch,in_ch,k)
            seq.append(sep)
        self.reside = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=(1,1)),
            nn.BatchNorm2d(out_ch),
        )
        last_sep = SeprationConv(in_ch,out_ch,k=k,last=True)
        seq.append(last_sep)
        self.seq = nn.Sequential(*seq)
        self.last_relu = nn.ReLU()

    def forward(self, x):
        start = x
        x = self.seq(x)
        res_out = self.reside(start)
        x = x + res_out
        x = self.last_relu(x)
        return x
    
class QuartNet(nn.Module):
    def __init__(self):
        super(QuartNet,self).__init__()
        self.first_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 33), stride=(1, 1),
                      padding=(0, 16)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.block1 = QuartNetBlock(repeat=3,in_ch=32,out_ch=128,k=(1,33))
        self.block2 = QuartNetBlock(repeat=3,in_ch=128,out_ch=128,k=(1,39))
        self.block3 = QuartNetBlock(repeat=3,in_ch=128,out_ch=128,k=(1,51))
    def forward(self, input):
        x = self.first_cnn(input)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x
class BatchLSTM(nn.Module):
    def __init__(self,in_ch=64,out_ch=512,\
            batch_f=True, bidirection=True):
        super().__init__()
        self.rnn = nn.LSTM(in_ch, out_ch, num_layers=1,
                    batch_first=batch_f, bidirectional=bidirection)
    
    def forward(self,x,length):
        x = nn.utils.rnn.pack_padded_sequence(x, enforce_sorted=False,
                                              lengths=length, batch_first=True)
        x, h = self.rnn(x)
        # x, h = self.rnn2(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)  # N*T*C
        return x

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=(8, 12), stride=(1, 1),
        #               padding=(4, 6)),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, (4, 8), (1, 1), (2, 4)),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        # )
        # self.maskcnn = MaskCNN()
        self.cnn = QuartNet()
        # x = math.ceil((float)(64-8+1+4*2))
        # x = math.ceil((float)(x-4+1+2*2))
        # self.rnn = nn.Sequential(
        #     nn.LSTM(64*x, 512, num_layers=1,
        #             batch_first=True, bidirectional=True),
        # )
        self.rnn = BatchLSTM(64*128,128,True,True)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc = nn.Linear(256, 29)

    def forward(self, input, percents):
        x = self.cnn(input)  # N*32*F*T
        # x = self.maskcnn(x, percents)
        x = x.view(x.size(0), x.size(1)*x.size(2), -1).transpose(1, 2).contiguous()
        # x = nn.utils.rnn.pack_padded_sequence(x, enforce_sorted=False,
        #                                       lengths=torch.mul(x.size(1), percents).int(), batch_first=True)
        # x, h = self.rnn(x)
        # # x, h = self.rnn2(x)
        # x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)  # N*T*C
        lengths=torch.mul(x.size(1), percents).int()
        x = self.rnn(x, lengths)
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
