from torch.utils import data
import torch
import os
import random
import glob
from os.path import join, basename, dirname, split
import numpy as np

# Below is the accent info for the used 10 speakers.
spk2acc = {'262': 'Edinburgh', #F
           '272': 'Edinburgh', #M
           '229': 'SouthEngland', #F 
           '232': 'SouthEngland', #M
           '292': 'NorthernIrishBelfast', #M 
           '293': 'NorthernIrishBelfast', #F 
           '360': 'AmericanNewJersey', #M
           '361': 'AmericanNewJersey', #F
           '248': 'India', #F
           '251': 'India'} #M
min_length = 256   # Since we slice 256 frames from each utterance when training.
# Build a dict useful when we want to get one-hot representation of speakers.
#speakers = ['p262', 'p272', 'p229', 'p232', 'p292', 'p293', 'p360', 'p361', 'p248', 'p251']
speakers = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']

spk2idx = dict(zip(speakers, range(len(speakers))))

def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    From Keras np_utils
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

class MyDataset(data.Dataset):
    """Dataset for MCEP, cwt_lf0 features and speaker labels."""
    def __init__(self, data_dir):
        data_files = glob.glob(join(data_dir, '*_f.npz'))
        data_files = [i for i in data_files if basename(i)[9:12] in speakers] 
        self.data_files = self.rm_too_short_utt(data_files)
        self.num_files = len(self.data_files)
        print("\t Number of training samples: ", self.num_files)
        for f in self.data_files:
            data = np.load(f)
            if data['normed_coded_sp'].shape[0] <= min_length:
                print(f)
                raise RuntimeError(f"The data may be corrupted! We need all MCEP features having more than {min_length} frames!") 

    def rm_too_short_utt(self, data_files, min_length=min_length):
        new_data_files = []
        for datafile in data_files:
            data = np.load(datafile)
            mc = data['normed_coded_sp']
            if mc.shape[0] > min_length:
                new_data_files.append(datafile)
        return new_data_files

    def sample_seg(self, feat, sample_len=min_length):
        assert feat.shape[0] - sample_len >= 0
        s = np.random.randint(0, feat.shape[0] - sample_len + 1)
        return feat[s:s+sample_len, :]

    def __len__(self):
        return self.num_files

    def __getitem__(self, index):
        filename = self.data_files[index]
        spk = basename(filename).split('_')[2]
        spk_idx = spk2idx[spk]
        data = np.load(filename)
        mc = self.sample_seg(data['normed_coded_sp'])
        lf0 = self.sample_seg(data['normed_cwt_lf0'])
        mc = np.transpose(mc, (1, 0))  # (T, D) -> (D, T), since pytorch need feature having shape
        lf0 = np.transpose(lf0, (1, 0))
        # to one-hot
        spk_cat = np.squeeze(to_categorical([spk_idx], num_classes=len(speakers)))

        return torch.FloatTensor(mc), torch.FloatTensor(lf0), torch.LongTensor([spk_idx]).squeeze_(), torch.FloatTensor(spk_cat)
        

class TestDataset(object):
    """Dataset for testing."""
    def __init__(self, data_dir, wav_dir, src_spk='p262', trg_spk='p272'):
        self.src_spk = src_spk
        self.trg_spk = trg_spk
        self.mc_files = sorted(glob.glob(join(data_dir, '{}*.npy'.format(self.src_spk))))

        self.src_spk_stats = np.load(join(data_dir.replace('test', 'train'), '{}_stats.npz'.format(src_spk)))
        self.trg_spk_stats = np.load(join(data_dir.replace('test', 'train'), '{}_stats.npz'.format(trg_spk)))
        
        self.logf0s_mean_src = self.src_spk_stats['log_f0s_mean']
        self.logf0s_std_src = self.src_spk_stats['log_f0s_std']
        self.logf0s_mean_trg = self.trg_spk_stats['log_f0s_mean']
        self.logf0s_std_trg = self.trg_spk_stats['log_f0s_std']
        self.mcep_mean_src = self.src_spk_stats['coded_sps_mean']
        self.mcep_std_src = self.src_spk_stats['coded_sps_std']
        self.mcep_mean_trg = self.trg_spk_stats['coded_sps_mean']
        self.mcep_std_trg = self.trg_spk_stats['coded_sps_std']
        self.src_wav_dir = f'{wav_dir}/{src_spk}'
        self.spk_idx = spk2idx[trg_spk]
        spk_cat = to_categorical([self.spk_idx], num_classes=len(speakers))
        self.spk_c_trg = spk_cat

    def get_batch_test_data(self, batch_size=8):
        batch_data = []
        for i in range(batch_size):
            mcfile = self.mc_files[i]
            filename = basename(mcfile).split('-')[-1]
            wavfile_path = join(self.src_wav_dir, filename.replace('npy', 'wav'))
            batch_data.append(wavfile_path)
        return batch_data       

def get_loader(data_dir, batch_size=32, mode='train', num_workers=1):
    dataset = MyDataset(data_dir)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers,
                                  drop_last=True)
    return data_loader


if __name__ == '__main__':
    loader = get_loader('./data/mc/train')
    data_iter = iter(loader)
    for i in range(10):
        mc, lf0, spk_idx, spk_acc_cat = next(data_iter)
        print('-'*50)
        print(mc.size())
        print(lf0.size())
        print(spk_idx.size())
        print(spk_acc_cat.size())
        print(spk_idx.squeeze_())
        print(spk_acc_cat)
        print('-'*50)