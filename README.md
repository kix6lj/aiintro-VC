# aiintro-VC

Code of aiintro group project.

## StarGAN-Voice-Conversion-master

### preprocess.py

将 data/VCTK-Corpus/wav48 里的所有音频文件降采样至 16 kHz，保存在 data/VCTK-Corpus/wav16 中。然后计算降采样后的音频的特征；作为一个 demo，preprocess.py 只计算其中十个音频的特征。

这里的特征以 .npy 扩展名保存，本身是一个 numpy 的 ndarray。

