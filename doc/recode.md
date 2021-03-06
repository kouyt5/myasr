# 实验结果记录

# train train-clean-100 
epoch0 eval avg cer:0.853
epoch1 eval avg cer:0.790
epoch6 eval avg cer:0.76
epoch7 eval avg cer:0.7587591505347183

epoch20
eval loss: 0.900949239730835
eval avg cer:0.37047709108989474
eval avg wer:0.8462259622323776
epoch21
eval loss: 0.8629829287528992
eval avg cer:0.3700133555155141
eval avg wer:0.8468222520871544

# train:dev-clean 原始cnn只有两层的模型，收敛较慢
epoch23 loss: 0.774954080581665
eval loss: 0.3025244176387787
eval avg cer:0.42139769648064995
eval avg wer:0.8993846360916363

epoch24 loss: 0.7303251624107361
eval loss: 0.3094230592250824
eval avg cer:0.4183025856189405
eval avg wer:0.9105837853743128

# train dev-clean  提高CNN的复杂度，仿照QuartNet模型，深度分离卷积未加group
epoch3 loss: 1.9922860860824585
eval loss: 1.4076844453811646
eval avg cer:0.6802419306478019
eval avg wer:0.9759158391615406

epoch4 loss: 1.8929121494293213
eval loss: 1.4002015590667725
eval avg cer:0.656932462319529
eval avg wer:0.9717026645399651
eval wer: 1.00 cer: 0.59

epoch5 loss: 1.8096849918365479
eval loss: 1.2609014511108398
eval avg cer:0.6215693910471338
eval avg wer:0.9687526974686644
eval wer: 1.00 cer: 0.45

epoch6 loss: 1.7204225063323975
eval loss: 1.1547366380691528
eval avg cer:0.6167796760518443
eval avg wer:0.9657506219738877
eval wer: 1.00 cer: 0.55

epoch7 loss: 1.658284068107605
eval loss: 1.0325913429260254
eval avg cer:0.5938753162718305
eval avg wer:0.9586368877691184
eval wer: 1.00 cer: 0.50

epoch8 loss: 1.6086034774780273
eval loss: 1.0615720748901367
eval avg cer:0.5783197672665522
eval avg wer:0.9633105247624025
eval wer: 1.00 cer: 0.50

epoch9 loss: 1.5483922958374023
eval loss: 0.9348485469818115
eval avg cer:0.5646703257424925
eval avg wer:0.9583989532267226
eval wer: 1.00 cer: 0.45
# train dev-clean  提高CNN的复杂度，仿照QuartNet模型，深度分离卷积加group
epoch8
eval loss: 1.3891783952713013
eval avg cer:0.535855003623429
eval avg wer:0.9640285785473611
# train dev-clean train100 同上
epoch1 loss: 1.8054180145263672
avg cer:0.6810618445554635
avg wer:0.9792982157125738
wer: 0.99 cer: 0.63

epoch1.5
eval loss: 2.1159555912017822
eval avg cer:0.6519714874949886
eval avg wer:0.9771847982990068

epoch6
eval loss: 1.5062522888183594
eval avg cer:0.5027759882667944
eval avg wer:0.9820793764291642
eval wer: 0.83 cer: 0.59

epoch7 loss: 0.947791576385498
avg cer:0.4323358179265553
avg wer:0.9037686874292445
wer: 0.87 cer: 0.37

epoch8
eval loss: 1.5154997110366821
eval avg cer:0.5569340197399835
eval avg wer:0.9766268180645884
eval wer: 1.00 cer: 0.68
epoch8 loss: 1.155289649963379
avg cer:0.41977440979674535
avg wer:0.8947762973825784
wer: 1.00 cer: 0.36

epoch9
eval loss: 1.2131375074386597
eval avg cer:0.48257271196422014
eval avg wer:0.9153084722409889
eval wer: 1.00 cer: 0.45
epoch10
eval loss: 1.1343376636505127
eval avg cer:0.4711898588191867
eval avg wer:0.9393172474519549
eval wer: 0.83 cer: 0.45

epoch12
eval loss: 1.3936150074005127
eval avg cer:0.46572216555982515
eval avg wer:0.9229052516228412
eval wer: 0.83 cer: 0.55

# train-100 QuartNet shuffle 在深度可分离卷积中间
epoch12 
eval loss: 0.8432261347770691
eval avg cer:0.3141713566294957
eval avg wer:0.6887609702796487
eval wer: 0.62 cer: 0.26
epoch13
eval loss: 0.7768691778182983
eval avg cer:0.2933178275210534
eval avg wer:0.681502759749911
eval wer: 0.67 cer: 0.28

epoch 47 wer=0.53

# train-100 QuartNet shuffle在深度可分离卷积后面 收敛较快
lr=0.01 weight_decay=1e-4 结构5*5
epoch0
eval avg loss: 1.8585794044772224
eval avg cer:0.5887520562397188
eval avg wer:0.9581412462394217

epoch1
eval avg loss: 1.3110620762820535
eval avg cer:0.4396990706003582
eval avg wer:0.8460822309307814

epoch2
eval avg loss: 1.3734615614902992
eval avg cer:0.42000609972551234
eval avg wer:0.8384373985499405
epoch3
eval avg loss: 1.1817917892817227
eval avg cer:0.37411003559857603
eval avg wer:0.8031992034459621
epoch5
eval avg loss: 0.8181729514460795
eval avg cer:0.26251675234532834
eval avg wer:0.6501352960277086
epoch6
eval avg loss: 0.7752706212616862
eval avg cer:0.2451388715282118
eval avg wer:0.6298043628808865
epoch7
eval avg loss: 0.7117968867175829
eval avg cer:0.22269210005398596
eval avg wer:0.5923185113058531
epoch8
eval avg loss: 0.709389779442726
eval avg cer:0.2788514908916347
eval avg wer:0.6061524473405061

9
eval avg loss: 0.7318803619051204
eval avg cer:0.22608560809560255
eval avg wer:0.5929197791490743
10
eval avg loss: 0.7358196154353391
eval avg cer:0.22273615993519386
eval avg wer:0.5937709672532087
11
eval avg loss: 0.7135171439259638
eval avg cer:0.23037931655035893
eval avg wer:0.5897602354366831
12
eval avg loss: 0.7544848580548834
eval avg cer:0.24835653760591897
eval avg wer:0.642169274383778
epoch40
wer0.53
如果加上audioAugment epoch22 wer=0.47 lr=0.1
# train100 augment 40 16 lr0.1 wd=1e-3
epoch 1
eval avg loss: 1.5675727397623196
eval avg cer:0.5044144261246039
eval avg wer:0.9065766409141076
epoch2
eval avg loss: 1.3738680002461763
eval avg cer:0.44557137929827895
eval avg wer:0.8766797221561032
# train100+360
wer0.33

# dist train-100

examples2_1  | eval avg loss: 0.7347967242224489
examples2_1  | eval avg cer:0.30134841800958295
examples2_1  | eval avg wer:0.6495117673804317
examples2_1  | trues: the physician who attended him was named terro he thought by some peculiar train of reasoning that he could cure him by applying a mercurial ointment to the chest to which no one raised any objection
examples2_1  | preds: e positieon who atenden him was the m terril he thought i some becu your tran o reasoning that he could cur him by apliing am mecurial otemento the cast to itch no an rased any ajection
  2%|▏         | 19/890 [00:35<27:11,  1.87s/it]epoch8avg loss: 0.7327426344156265

分布式训练不能达到单机训练的精度，考虑同步bn和warmup