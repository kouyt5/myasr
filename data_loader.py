import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import os
# import h5py

class MyAudioDataset(Dataset):
    def __init__(self, manifest_path, labels_path, max_duration=16,mask=False):
        self.mask = mask
        self.datasets = []
        self.labels = {}
        with open(manifest_path, encoding='utf-8') as f:
            for line in f.readlines():
                data = json.loads(line, encoding='utf-8')
                if data['duration'] > max_duration:
                    continue
                self.datasets.append(data)
        with open(labels_path, encoding='utf-8') as f:
            idx = [char.replace('\n', '') for char in f.readlines()]
            self.index2char = dict([(i, idx[i]) for i in range(len(idx))])
            self.char2index = dict([(idx[i], i) for i in range(len(idx))])

    def __getitem__(self, index):
        data = self.datasets[index]
        text2id = [self.char2index[char] for char in data['text']]
        return self.parse_audio(data["audio_filepath"],mask=self.mask), text2id, data['audio_filepath']

    def parse_audio(self, audio_path, win_len=0.02,mask=False):
        if not os.path.exists(path=audio_path):
            raise("音频路径不存在 "+ audio_path)
        y, sr = torchaudio.load(audio_path)
        n_fft = int(win_len * sr)
        hop_length = n_fft // 2
        transformer = torchaudio.transforms.MelSpectrogram(sr, n_fft=n_fft,
                                                           hop_length=hop_length, n_mels=64)
        spec = transformer(y)
        y = torchaudio.transforms.AmplitudeToDB(stype="power")(spec)

        # F-T mask
        audio_f_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)
        audio_t_mask = torchaudio.transforms.TimeMasking(time_mask_param=30)
        if mask:
            y = audio_f_mask(y)
            y = audio_t_mask(y)
        # 归一化
        std, mean = torch.std_mean(y)
        y = torch.div((y-mean), std)
        return y  # (1,64,T)

    def id2txt(self, id_list):
        """
        根据id获取对应的文本
        :params id_list id的列表[1,3,...]
        """
        for id in id_list:
            if id >= len(self.index2char):
                raise "index out of the lengths请检查id的大小范围"
        return ''.join([self.index2char[id] for id in txt2id])
    def __len__(self):
        return len(self.datasets)

class MyAudioLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(MyAudioLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn

    def _collate_fn(self, batch):
        batch = sorted( # batch_size * {(1,64,T),[1,4,...],duration}
            batch, key=lambda sample: sample[0].size(2), reverse=True) 
        longest_sample = max(batch, key=lambda x: x[0].size(2))[0] # (1,64,T)
        freq_size = longest_sample.size(1)
        minibatch_size = len(batch)
        max_seqlength = longest_sample.size(2) # 时域长度
        max_trans_length = len(max(batch, key=lambda x: len(x[1]))[1]) # 文本长度
        inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
        input_percentages = torch.FloatTensor(minibatch_size) # mask 使用
        target_sizes = torch.IntTensor(minibatch_size) # 
        # targets = torch.zeros(minibatch_size,max_trans_length,dtype=torch.int16)
        targets = []
        paths = []
        for x in range(minibatch_size):
            sample = batch[x]
            tensor = sample[0].squeeze(0) # (64,T)
            trans_txt = sample[1]
            audio_path = sample[2]
            paths.append(audio_path)
            seq_length = tensor.size(1) # 时域长度T
            inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
            input_percentages[x] = seq_length / float(max_seqlength)
            target_sizes[x] = len(trans_txt)
            # targets[x].narrow(0,0,len(trans_txt)).copy_(torch.IntTensor(trans_txt))
            targets.extend(trans_txt)
        # return [(N,64,T),(N,Length),(N),(N),lists] N=batch_size
        targets = torch.IntTensor(targets)
        return inputs, targets, input_percentages, target_sizes, paths

class APCDataset(Dataset):
    def __init__(self, manifest_path, labels_path, h5_file_path=None, max_duration=16,mask=False):
        self.mask = mask
        self.datasets = []
        self.labels = {}
        with open(manifest_path, encoding='utf-8') as f:
            for line in f.readlines():
                data = json.loads(line, encoding='utf-8')
                if data['duration'] > max_duration:
                    continue
                audio_name = data['audio_filepath'].split('/')[-1].split('.')[0] + ".pt"
                data['audio_filepath'] = os.path.join(h5_file_path,audio_name)
                self.datasets.append(data)
        with open(labels_path, encoding='utf-8') as f:
            idx = [char.replace('\n', '') for char in f.readlines()]
            self.index2char = dict([(i, idx[i]) for i in range(len(idx))])
            self.char2index = dict([(idx[i], i) for i in range(len(idx))])

    def __getitem__(self, index):
        data = self.datasets[index]
        text2id = [self.char2index[char] for char in data['text']]
        return self.parse_audio(data["audio_filepath"]), text2id, data['audio_filepath']

    def parse_audio(self, audio_path):
        if not os.path.exists(path=audio_path):
            print(audio_path)
            raise("文件路径不存在 ")
        y = torch.load(audio_path)
        y = torch.transpose(y,1,2)
        return y
        # with h5py.File(file_path,'r') as f:
        #     data = f['features']
        #     y = torch.Tensor(data.value)

        return y  # (1,512,T)
    
    def __len__(self):
        return len(self.datasets)

if __name__ == "__main__":
    manifest_path = "./data/dev-clean.json"
    labels_path = "./data/labels.txt"
    h5_path = "/mnt/volume/workspace/Autoregressive-Predictive-Coding/extra/dev_clean"
    dataset = MyAudioDataset(manifest_path, labels_path)
    apcdatasets = APCDataset(manifest_path, labels_path,h5_path)
    # datasets测试
    data = dataset.__getitem__(1)
    data2 = apcdatasets.__getitem__(1)
    txt2id = data[1]
    id2txt = dataset.id2txt(txt2id)
    # dataloader 测试
    dataloader = MyAudioLoader(apcdatasets,shuffle=True,batch_size=8)
    for batch in enumerate(dataloader):
        print("inputs:" + str(batch[1][0].size()))
        print("targets:" + str(batch[1][1].size()))
        print("input_percentages:" + str(batch[1][2].size()))
        print("target_sizes:" + str(batch[1][3].size()))
        print(len(batch[1][4]))
