# myasr
## 语音识别模型0基础搭建
参考 [deepspeech-pytorch](https://github.com/SeanNaren/deepspeech.pytorch) 中的代码，完成语音识别模型的一步步搭建。

## 目的
之前做的语音识别大都是在别人的模型基础上调参改进，虽然熟悉语音识别模型搭建的必须组件，但是对于具体的细节却是不敢说完全了解，因此想通过自行构建一个语音识别模型,来深入了解语音识别模型构建需要注意的地方。首先是实现能够在Librispeech dev-clean数据集上实现过拟合。以做为研究语音识别的基础。

## 前期已完成
+ 数据加载`data_loader.py` 完全使用`torchaudio` 库实现特征提取。
+ 模型构建，仿造deepspeech2模型，构建CNN+LSTM模型。
+ 解码，首先实现贪婪搜索解码，不使用语言模型，该部分主要参考 deepspeech

该部分发布在release中，作为存档
## 后期计划

- [x] 模型增加复杂度，使用全数据集训练
- [x] 添加语言模型支持

## 坑
+ 数据预处理时没有对语音特征取对数，造成特征值过小模型无法训练
## v1
epoch35:avg cer=0.328
epoch26avg cer=0.315
epoch27avg cer=0.302
epoch28avg cer=0.288
epoch29avg cer=0.279
epoch30avg cer:0.264
epoch31avg cer=0.248
epoch32avg cer:0.233
epoch33avg cer=0.219
epoch35avg cer=0.202
epoch36avg cer=0.192
epoch37avg cer=0.180
epoch38avg cer=0.172
epoch39avg cer=0.159