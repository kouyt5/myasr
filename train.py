import torch
import os,sys
from MyModel import MyModel, MyModel2
from data_loader import MyAudioLoader, MyAudioDataset
import torch.nn as nn
from decoder import GreedyDecoder
from tqdm import tqdm
from ASR_metrics import utils as metrics
from apex import amp
from torchsummary import summary
# from torch.nn.parallel import DistributedDataParallel
from apex.parallel import DistributedDataParallel
from torchelastic.utils.data import ElasticDistributedSampler
import torch.distributed as dist
from datetime import timedelta
from ruamel.yaml import YAML

def set_lr(optimizer,lr,weigth_decay):
    for param in optimizer.param_groups:
        param['lr'] = lr
        param['weight_decay']=weigth_decay
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
    return total_loss/total_count

# dist
device_id = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(device_id)
print(f"=> set cuda device = {device_id}")
os.environ["NCCL_BLOCKING_WAIT"] = "1"
dist.init_process_group(
    backend="nccl", init_method="env://", timeout=timedelta(seconds=10)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_path = "conf.yaml"
with open(config_path, encoding='utf-8') as f:
    params = YAML(typ='safe').load(f)
dev_manifest_path = params['datasets']['dev_datasets']
train_manifest_path = params['datasets']['train_datasets']
labels_path = params['datasets']['label']
model = MyModel2()

# 使用Adam无法收敛，SGD比较好调整
optim = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True,weight_decay=1e-4)
# optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5, amsgrad=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,"min",\
        factor=0.1,patience=2,min_lr=1e-4,verbose=True)
# apex 混合精度加速训练
opt_level = 'O1'
model, optim = amp.initialize(model, optim, opt_level=opt_level)
# dist
# model = DistributedDataParallel(model, device_ids=[device_id])
model = DistributedDataParallel(model)
model.to(device)
dev_datasets = MyAudioDataset(dev_manifest_path, labels_path)
dev_dataloader = MyAudioLoader(dev_datasets, batch_size=4, drop_last=True,shuffle=True)
train_datasets = MyAudioDataset(train_manifest_path, labels_path,max_duration=17,mask=True)
train_sampler = ElasticDistributedSampler(train_datasets)
train_dataloader = MyAudioLoader(train_datasets, batch_size=32, drop_last=True,sampler=train_sampler)
criterion = nn.CTCLoss(blank=0, reduction="mean")
decoder = GreedyDecoder(labels_path)

# model = model.to(device=device)
summary(model,[(64,512),(1,)],device="cuda") # 探测模型结构

# checkpoint = torch.load('checkpoint/0.pt')
# model.load_state_dict(checkpoint['model'])
# optim.load_state_dict(checkpoint['optimizer'])
# amp.load_state_dict(checkpoint['amp'])
# scheduler.load_state_dict(checkpoint['scheduler'])
# evalute(model, dev_dataloader, device)
# set_lr(optim,0.01,1e-4)
for epoch in range(0, 60):
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
        ground_trues = []
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
        total_loss += loss.item()
        if total_count % 5 == 0:
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
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optim.state_dict(),
                'amp': amp.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            torch.save(checkpoint, 'checkpoint/{}.pt'.format(epoch))
    avg_loss = evalute(model, dev_dataloader, device)
    scheduler.step(avg_loss)
