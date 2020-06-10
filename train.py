import torch
from MyModel import MyModel, MyModel2
from data_loader import MyAudioLoader, MyAudioDataset
import torch.nn as nn
from decoder import GreedyDecoder
from tqdm import tqdm
from ASR_metrics import utils as metrics
from apex import amp


def evalute(model, loader, device):
    model.eval()
    model.to(device=device)
    cer_list_pairs = []
    wer_list_pairs = []
    total_count = 0
    total_loss = .0
    print("eval...")
    for (i, batch) in tqdm(enumerate(loader)):
        input = batch[0]
        percents = batch[2]
        trans = batch[1]
        trans_lengths = batch[3]
        out = model(input.to(device), percents.to(device))
        t_lengths = torch.mul(out.size(1), percents).int()  # 输出实际长度
        loss = criterion(out.transpose(0, 1),
                         trans, t_lengths, trans_lengths)
        trans_pre = decoder.decode(out)  # 预测文本
        ground_trues = []  # 实际文本
        start = 0
        for i in range(len(trans_lengths)):
            add = trans_lengths[i].item()
            trans_list = trans.narrow(0, start, add).numpy().tolist()
            start += add
            ground_trues.append(decoder.decoder_by_number(trans_list))
        cer_list_pairs.extend([(ground_trues[i].replace(' ', ''), trans_pre[0][i][0].replace(' ', ''))
                          for i in range(len(trans_lengths))])
        wer_list_pairs.extend([(ground_trues[i], trans_pre[0][i][0])
                          for i in range(len(trans_lengths))])
        total_count += 1
        total_loss += loss.item()
    try:
        wer = metrics.compute_wer_list_pair(wer_list_pairs)
        cer = metrics.calculate_cer_list_pair(cer_list_pairs)
    except ZeroDivisionError:
        print('ZeroDivisionError')
    print("eval avg loss: "+str(total_loss/total_count))
    print("eval avg cer:{}".format(cer, '0.2f'))
    print("eval avg wer:{}".format(wer, '0.2f'))
    print("trues: "+ground_trues[0])
    print("preds: "+trans_pre[0][0][0])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dev_manifest_path = "./data/dev-clean.json"
train_manifest_path = "./data/train-clean-100.json"
labels_path = "./data/labels.txt"
model = MyModel2()

dev_datasets = MyAudioDataset(dev_manifest_path, labels_path)
dev_dataloader = MyAudioLoader(dev_datasets, batch_size=4, drop_last=True,shuffle=True)
train_datasets = MyAudioDataset(train_manifest_path, labels_path,max_duration=16)
train_dataloader = MyAudioLoader(train_datasets, batch_size=16, drop_last=True,shuffle=True)
criterion = nn.CTCLoss(blank=0, reduction="mean")
optim = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, nesterov=True,weight_decay=1e-4)
# optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5, amsgrad=True)
decoder = GreedyDecoder(labels_path)
if torch.cuda.is_available():
    map_location = 'cuda:0'
else:
    map_location = 'cpu'
# model = torch.load('checkpoint/0.pth')
# evalute(model, dev_dataloader, device)

# apex
model.to(device=device)
opt_level = 'O1'
model, optim = amp.initialize(model, optim, opt_level=opt_level)
checkpoint = torch.load('checkpoint/17.pt')
model.load_state_dict(checkpoint['model'])
optim.load_state_dict(checkpoint['optimizer'])
amp.load_state_dict(checkpoint['amp'])
evalute(model, dev_dataloader, device)
for epoch in range(18, 150):
    # torch.save(model, "checkpoint/"+str(epoch)+".pt")
    cer_list_pairs = []
    wer_list_pairs = []
    total_count = 0
    total_loss = 0
    model.train()
    for (i, batch) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        total_count +=1
        input = batch[0]
        percents = batch[2]
        trans = batch[1]
        trans_lengths = batch[3]
        out = model(input.to(device), percents.to(device))
        t_lengths = torch.mul(out.size(1), percents).int()  # 输出实际长度
        loss = criterion(out.transpose(0, 1),
                         trans, t_lengths, trans_lengths)
        optim.zero_grad()
        with amp.scale_loss(loss, optim) as scaled_loss:
            scaled_loss.backward()
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
        cer_list_pairs.extend([(ground_trues[i].replace(' ', ''), trans_pre[0][i][0].replace(' ', ''))
                          for i in range(len(trans_lengths))])
        wer_list_pairs.extend([(ground_trues[i], trans_pre[0][i][0])
                          for i in range(len(trans_lengths))])
        total_loss += loss.item()
        if total_count % 50 == 0:
            try:
                wer = metrics.compute_wer_list_pair(wer_list_pairs)
                cer = metrics.calculate_cer_list_pair(cer_list_pairs)
            except ZeroDivisionError:
                print('ZeroDivisionError')
                continue
            print("epoch"+str(epoch) + "avg loss: "+str(total_loss/total_count))
            print("avg cer:{}".format(cer, '0.2f'))
            print("avg wer:{}".format(wer, '0.2f'))
            print("trues: "+ground_trues[0])
            print("preds: "+trans_pre[0][0][0])
    # torch.save(model, "checkpoint/"+str(epoch)+".pth")
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optim.state_dict(),
                'amp': amp.state_dict()
            }
            torch.save(checkpoint, 'checkpoint/{}.pt'.format(epoch))
    evalute(model, dev_dataloader, device)
