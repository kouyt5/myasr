import torch
from MyModel import MyModel
from data_loader import MyAudioLoader, MyAudioDataset
import torch.nn as nn
from decoder import GreedyDecoder
from tqdm import tqdm
from ASR_metrics import utils as metrics


def evalute(model, loader, device):
    model.eval()
    total_cer = 0
    total_wer = 0
    total_count = 0
    print("eval...")
    for (i, batch) in tqdm(enumerate(loader)):
        model.to(device=device)
        input = batch[0]
        percents = batch[2]
        trans = batch[1]
        trans_lengths = batch[3]
        out = model(input.to(device), percents.to(device))
        t_lengths = torch.mul(out.size(1), percents).int()  # 输出实际长度
        loss = criterion(out.transpose(0, 1).requires_grad_(),
                         trans, t_lengths, trans_lengths)
        trans_pre = decoder.decode(out)  # 预测文本
        ground_trues = []  # 实际文本
        start = 0
        for i in range(len(trans_lengths)):
            add = trans_lengths[i].item()
            trans_list = trans.narrow(0, start, add).numpy().tolist()
            start += add
            ground_trues.append(decoder.decoder_by_number(trans_list))
        cer_list_pairs = [(trans_pre[0][i][0], ground_trues[i])
                          for i in range(len(trans_lengths))]
        try:
            wer = metrics.compute_wer_list_pair(cer_list_pairs)
            cer = metrics.calculate_cer_list_pair(cer_list_pairs)
        except ZeroDivisionError:
            print('ZeroDivisionError')
            continue
        total_cer += cer
        total_wer += wer
        total_count += 1
        # print(ground_trues[0])
        # print(trans_pre[0][0][0])
    print("eval loss: "+str(loss.item()))
    print("eval avg cer:{}".format(total_cer/total_count))
    print("eval avg wer:{}".format(total_wer/total_count))
    print("eval wer: {} cer: {}".format(
        format(wer, '0.2f'), format(cer, '0.2f')))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dev_manifest_path = "./data/dev-clean.json"
train_manifest_path = "./data/train-clean-100.json"
labels_path = "./data/labels.txt"
model = MyModel()

dev_datasets = MyAudioDataset(dev_manifest_path, labels_path)
dev_dataloader = MyAudioLoader(dev_datasets, batch_size=32, drop_last=False)
train_datasets = MyAudioDataset(train_manifest_path, labels_path)
train_dataloader = MyAudioLoader(train_datasets, batch_size=32, drop_last=True)
criterion = nn.CTCLoss(blank=0, reduction="mean")
optim = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
decoder = GreedyDecoder(labels_path)
if torch.cuda.is_available():
    map_location = 'cuda:0'
else:
    map_location = 'cpu'
# model = torch.load('checkpoint/0.pth')
# evalute(model, dev_dataloader, device)
for epoch in range(150):
    torch.save(model, "checkpoint/"+str(epoch)+".pth")
    total_cer = 0
    total_wer = 0
    total_count = 0
    model.train()
    for (i, batch) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        optim.zero_grad()
        model.to(device=device)
        input = batch[0]
        percents = batch[2]
        trans = batch[1]
        trans_lengths = batch[3]
        out = model(input.to(device), percents.to(device))
        t_lengths = torch.mul(out.size(1), percents).int()  # 输出实际长度
        loss = criterion(out.transpose(0, 1).requires_grad_(),
                         trans, t_lengths, trans_lengths)
        loss.backward()
        optim.step()
        trans_pre = decoder.decode(out)
        # print(trans_pre[0][0][0])
        ground_trues = []
        start = 0
        for i in range(len(trans_lengths)):
            add = trans_lengths[i].item()
            trans_list = trans.narrow(0, start, add).numpy().tolist()
            start += add
            ground_trues.append(decoder.decoder_by_number(trans_list))
        # true = trans.narrow(0,0,trans_lengths[0].int()).int().numpy().tolist()
        # print(ground_trues[0])
        cer_list_pairs = [(trans_pre[0][i][0], ground_trues[i])
                          for i in range(len(trans_lengths))]
        try:
            wer = metrics.compute_wer_list_pair(cer_list_pairs)
            cer = metrics.calculate_cer_list_pair(cer_list_pairs)
        except ZeroDivisionError:
            print('ZeroDivisionError')
            continue
        total_cer += cer
        total_wer += wer
        total_count += 1
        if total_count % 50 == 0:
            print("epoch"+str(epoch) + " loss: "+str(loss.item()))
            print("avg cer:{}".format(total_cer/total_count))
            print("avg wer:{}".format(total_wer/total_count))
            print("wer: {} cer: {}".format(
                format(wer, '0.2f'), format(cer, '0.2f')))
            print(ground_trues[0])
            print(trans_pre[0][0][0])
    torch.save(model, "checkpoint/"+str(epoch)+".pth")
    evalute(model, dev_dataloader, device)
