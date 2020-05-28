import torch
from MyModel import MyModel
from data_loader import MyAudioLoader, MyAudioDataset
import torch.nn as nn
from decoder import GreedyDecoder
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
manifest_path = "./data/dev-clean.json"
labels_path = "./data/labels.txt"
model = MyModel()

datasets = MyAudioDataset(manifest_path,labels_path)
dataloader = MyAudioLoader(datasets,batch_size=8,drop_last=True)
criterion = nn.CTCLoss(blank=0,reduction="mean")
optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
decoder = GreedyDecoder(labels_path)
# model = torch.load('checkpoint/0.pth')
for epoch in range(20):
    torch.save(model,"checkpoint/"+str(epoch)+".pth")
    for (i,batch) in tqdm(enumerate(dataloader)):
        model.train()  
        optim.zero_grad()
        model.to(device=device)
        input = batch[0]
        percents = batch[2]
        trans = batch[1]
        trans_lengths = batch[3]
        out = model(input, percents)

        t_lengths = torch.mul(out.size(1),percents).int() # 输出实际长度
        loss = criterion(out.transpose(0,1).requires_grad_(),trans,t_lengths,trans_lengths)
        loss.backward()
        optim.step()
        print("epoch"+str(epoch)+"loss"+str(loss.data))
        trans_pre = decoder.decode(out)
        print(trans_pre[0][0])
        true = trans.narrow(0,0,trans_lengths[0].int()).int().numpy().tolist()
        print(decoder.decoder_by_number(true))
    torch.save(model,"checkpoint/"+str(epoch)+".pth")
