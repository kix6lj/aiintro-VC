from model import Generator, Generatorf0
from utils import *
import torch
import numpy as np

speakers = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']
spk2idx = dict(zip(speakers, range(len(speakers))))

class Converter():
    def __init__(self, G_path, Gf0_path, stats_path):
        '''
            Initialize and load model
            model are stored in 'model_dir'
        '''
        self.sampling_rate = 16000
        self.num_mceps = 36
        self.device = 'cpu'
        self.G = Generator(num_speakers=6)
        self.Gf0 = Generatorf0(num_speakers=6, scale=10)d
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.Gf0.load_state_dict(torch.load(Gf0_path, map_location=lambda storage, loc: storage))
        self.stats_path = stats_path
        
        self.G.to(self.device)
        self.Gf0.to(self.device)
        
    def convert(self, f0, timeaxis, sp, ap, to_class):
        '''
            Give f0, timeaxis, sp, ap (get these arugments directly by world_decompose in utils)
            Reture converted f0, timeaxis, coded_sp, ap (put these arguments into world_speech_synthesis in utils)
        '''
        target_stats = np.load(os.path.join(self.stats_path, '{}_stats.npz'.format(to_class)))
        log_f0s_mean_trg, log_f0s_std_trg = target_stats['log_f0s_mean'], target_stats['log_f0s_std']
        mcep_mean_trg, mcep_std_trg = target_stats['coded_sps_mean'], target_stats['coded_sps_std']
        mcep_mean_trg, mcep_std_trg = np.expand_dims(mcep_mean_trg, axis=-1), np.expand_dims(mcep_std_trg, axis=-1)
        # get target categorical
        spkidx = spk2idx[to_class]
        spk_cat = np.zeros(6)
        spk_cat[spkidx] = 1
    
        # coded_sp
        coded_sp = world_encode_spectral_envelop(sp = sp, fs = self.sampling_rate, dim = self.num_mceps)
        coded_sp_transposed = coded_sp.T # (time, dim) to (dim, time)
        mcep_mean_org, mcep_std_org = np.mean(coded_sp_transposed, axis=-1), np.std(coded_sp_transposed, axis=-1)
        mcep_mean_org, mcep_std_org = np.expand_dims(mcep_mean_org, axis=-1), np.expand_dims(mcep_std_org, axis=-1)
        coded_sp_norm = (coded_sp_transposed - mcep_mean_org) / mcep_std_org
        
        # Wavelet lf0
        uv, cont_lf0_lpf = get_cont_lf0(f0)
        logf0s_mean_org, logf0s_std_org = np.mean(cont_lf0_lpf), np.std(cont_lf0_lpf)
        cont_lf0_lpf_norm = (cont_lf0_lpf - logf0s_mean_org) / logf0s_std_org
        Wavelet_lf0, scales = get_lf0_cwt(cont_lf0_lpf_norm)
        
        Wavelet_lf0_norm, mean, std = norm_scale(Wavelet_lf0) 
        lf0_cwt_norm = Wavelet_lf0_norm.T 
        
        # additional process for the model
        input_sp = torch.FloatTensor(coded_sp_norm)
        input_lf0 = torch.FloatTensor(lf0_cwt_norm)
        input_cat = torch.FloatTensor(spk_cat)
        input_sp = input_sp.to(self.device)
        input_lf0 = input_lf0.to(self.device)
        input_cat = input_cat.to(self.device)
        
        input_sp.unsqueeze_(0)
        input_sp.unsqueeze_(0)
        input_lf0.unsqueeze_(0)
        input_cat.unsqueeze_(0)
        
        # feed into model
        coded_sp_converted_norm = self.G(input_sp, input_cat).data.cpu().numpy()
        lf0 = self.Gf0(input_lf0, input_cat).data.cpu().numpy()
        coded_sp_converted_norm, lf0 = np.squeeze(coded_sp_converted_norm, axis=(0, 1)), np.squeeze(lf0, axis=(0))
        
        ########## Recover ###############
        coded_sp_converted = coded_sp_converted_norm * mcep_std_trg + mcep_mean_trg #mceps

        lf0_cwt_denormalize = denormalize(lf0.T, mean, std)
        lf0_rec = inverse_cwt(lf0_cwt_denormalize, scales)
        lf0_converted = lf0_rec * log_f0s_std_trg + log_f0s_mean_trg
        f0_converted = np.squeeze(uv) * np.exp(lf0_converted)
        f0_converted = np.ascontiguousarray(f0_converted)
        
        coded_sp_converted = coded_sp_converted.T
        coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
        
        return f0_converted, timeaxis, coded_sp_converted, ap