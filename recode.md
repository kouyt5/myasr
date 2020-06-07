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