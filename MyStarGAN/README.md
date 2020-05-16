# MyStarGAN

这是对 pytorch 版的参考资料代码的修改。

### preprocess.py

将 data/raw/wav48 里的所有音频文件降采样至 16 kHz（但原式音频文件不需要是 48 kHz），保存在 data/raw/wav16 中。然后计算降采样后的音频的特征（是什么特征暂时没探究），保存在 data/mc/train 和 data/mc/test 中。

特征以 .npy 扩展名保存，是一个 numpy 的 ndarray，可以用 `np.load` 加载。

作为一个 demo，preprocess.py 只处理两个说话人的音频。考虑到音频文件和特征文件都会很大，因此整个 data 文件夹都被添加到了 .gitignore 中，需要自己手动添加这两个说话人并运行 preprocess.py 生成数据。添加说话人的方法是将 VCC 的数据里的 SF1 和 SM1 文件夹放到 data/raw/wav48 中，如果要添加其他说话人，还需要更改 preprocess.py 的代码。