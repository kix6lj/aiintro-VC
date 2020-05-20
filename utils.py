import librosa
import numpy as np
import os
import pyworld
from pycwt import wavelet
import pywt
from sklearn import preprocessing
from scipy.interpolate import interp1d
from scipy.signal import firwin, lfilter

def load_wav(wav_file, sr):
    wav, _ = librosa.load(wav_file, sr=sr, mono=True)
    return wav

def convert_continuos_f0(f0):
    '''CONVERT F0 TO CONTINUOUS F0
    Args:
        f0 (ndarray): original f0 sequence with the shape (T)
    Return:
        (ndarray): continuous f0 with the shape (T)
    '''
    # get uv information as binary
    uv = np.float32(f0 != 0)

    # get start and end of f0
    if (f0 == 0).all():
        logging.warn("all of the f0 values are 0.")
        return uv, f0
    start_f0 = f0[f0 != 0][0]
    end_f0 = f0[f0 != 0][-1]

    # padding start and end of f0 sequence
    start_idx = np.where(f0 == start_f0)[0][0]
    end_idx = np.where(f0 == end_f0)[0][-1]
    f0[:start_idx] = start_f0
    f0[end_idx:] = end_f0

    # get non-zero frame index
    nz_frames = np.where(f0 != 0)[0]

    # perform linear interpolation
    f = interp1d(nz_frames, f0[nz_frames])
    cont_f0 = f(np.arange(0, f0.shape[0]))

    return uv, cont_f0

def get_cont_lf0(f0, frame_period=5.0):
    uv, cont_f0_lpf = convert_continuos_f0(f0)
    #cont_f0_lpf = low_pass_filter(cont_f0_lpf, int(1.0 / (frame_period * 0.001)), cutoff=20)
    cont_lf0_lpf = np.log(cont_f0_lpf)
    return uv, cont_lf0_lpf

def get_lf0_cwt(lf0):
    '''
    input: 
        signal of shape (N)
    output: 
        Wavelet_lf0 of shape(10, N), scales of shape(10)
    '''
    mother = wavelet.MexicanHat()
    dt = 0.005
    dj = 1
    s0 = dt * 2
    J = 9
    
    Wavelet_lf0, scales, _, _, _, _ = wavelet.cwt(np.squeeze(lf0), dt, dj, s0, J, mother)
    # Wavelet.shape => (J + 1, len(lf0))
    Wavelet_lf0 = np.real(Wavelet_lf0).T
    return Wavelet_lf0, scales

def norm_scale(Wavelet_lf0):
    Wavelet_lf0_norm = np.zeros((Wavelet_lf0.shape[0], Wavelet_lf0.shape[1]))
    mean = np.zeros((1,Wavelet_lf0.shape[1]))#[1,10]
    std = np.zeros((1, Wavelet_lf0.shape[1]))
    for scale in range(Wavelet_lf0.shape[1]):
        mean[:,scale] = Wavelet_lf0[:,scale].mean()
        std[:,scale] = Wavelet_lf0[:,scale].std()
        Wavelet_lf0_norm[:,scale] = (Wavelet_lf0[:,scale]-mean[:,scale])/std[:,scale]
    return Wavelet_lf0_norm, mean, std

def normalize_cwt_lf0(f0, mean, std):
    uv, cont_lf0_lpf = get_cont_lf0(f0)
    cont_lf0_norm = (cont_lf0_lpf - mean) / std
    Wavelet_lf0, scales = get_lf0_cwt(cont_lf0_norm)
    Wavelet_lf0_norm, _, _ = norm_scale(Wavelet_lf0)
    
    return Wavelet_lf0_norm

def get_lf0_cwt_norm(f0s, mean, std):

    uvs = list()
    cont_lf0_lpfs = list()
    cont_lf0_lpf_norms = list()
    Wavelet_lf0s = list()
    Wavelet_lf0s_norm = list()
    scaless = list()

    means = list()
    stds = list()
    for f0 in f0s:

        uv, cont_lf0_lpf = get_cont_lf0(f0)
        cont_lf0_lpf_norm = (cont_lf0_lpf - mean) / std 

        Wavelet_lf0, scales = get_lf0_cwt(cont_lf0_lpf_norm) #[560,10]
        Wavelet_lf0_norm, mean_scale, std_scale = norm_scale(Wavelet_lf0) #[560,10],[1,10],[1,10]

        Wavelet_lf0s_norm.append(Wavelet_lf0_norm)
        uvs.append(uv)
        cont_lf0_lpfs.append(cont_lf0_lpf)
        cont_lf0_lpf_norms.append(cont_lf0_lpf_norm)
        Wavelet_lf0s.append(Wavelet_lf0)
        scaless.append(scales)
        means.append(mean_scale)
        stds.append(std_scale)

    return Wavelet_lf0s_norm, scaless, means, stds

def inverse_cwt(Wavelet_lf0, scales):
    '''
    Recovering signal
    input: 
        Wavelet_lf0 of shape(N, J + 1), scales of shape(J + 1)
    output:
        signal of shape(N)
    '''
    lf0_rec = np.zeros([Wavelet_lf0.shape[0], len(scales)])
    for i in range(0, len(scales)):
        lf0_rec[:,i] = Wavelet_lf0[:,i] * ((i+1+2.5)**(-2.5))
    lf0_rec_sum = np.sum(lf0_rec, axis = 1)
    lf0_rec_sum = preprocessing.scale(lf0_rec_sum)
    return lf0_rec_sum

def world_decompose(wav, fs, frame_period = 5.0):
    # Decompose speech signal into f0, spectral envelope and aperiodicity using WORLD
    wav = wav.astype(np.float64)
    f0, timeaxis = pyworld.harvest(wav, fs, frame_period = frame_period, f0_floor = 71.0, f0_ceil = 800.0)
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs)
    ap = pyworld.d4c(wav, f0, timeaxis, fs)
    return f0, timeaxis, sp, ap

def world_encode_spectral_envelop(sp, fs, dim=36):
    # Get Mel-cepstral coefficients (MCEPs)
    #sp = sp.astype(np.float64)
    coded_sp = pyworld.code_spectral_envelope(sp, fs, dim)
    return coded_sp

def world_decode_spectral_envelop(coded_sp, fs):
    # Decode Mel-cepstral to sp
    fftlen = pyworld.get_cheaptrick_fft_size(fs)
    decoded_sp = pyworld.decode_spectral_envelope(coded_sp, fs, fftlen)
    return decoded_sp

def world_encode_wav(wav_file, fs, frame_period=5.0, coded_dim=36):
    wav = load_wav(wav_file, sr=fs)
    f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=fs, frame_period=frame_period)
    coded_sp = world_encode_spectral_envelop(sp = sp, fs = fs, dim = coded_dim)
    return f0, timeaxis, sp, ap, coded_sp

def world_speech_synthesis(f0, coded_sp, ap, fs, frame_period):
    decoded_sp = world_decode_spectral_envelop(coded_sp, fs)
    # TODO
    min_len = min([len(f0), len(coded_sp), len(ap)])
    f0 = f0[:min_len]
    coded_sp = coded_sp[:min_len]
    ap = ap[:min_len]
    wav = pyworld.synthesize(f0, decoded_sp, ap, fs, frame_period)
    # Librosa could not save wav if not doing so
    wav = wav.astype(np.float32)
    return wav

def world_synthesis_data(f0s, coded_sps, aps, fs, frame_period):
    wavs = list()
    for f0, decoded_sp, ap in zip(f0s, coded_sps, aps):
        wav = world_speech_synthesis(f0, coded_sp, ap, fs, frame_period)
        wavs.append(wav)
    return wavs

def coded_sps_normalization_fit_transoform(coded_sps):
    coded_sps_concatenated = np.concatenate(coded_sps, axis = 1)
    coded_sps_mean = np.mean(coded_sps_concatenated, axis = 1, keepdims = True)
    coded_sps_std = np.std(coded_sps_concatenated, axis = 1, keepdims = True)
    coded_sps_normalized = list()
    for coded_sp in coded_sps:
        coded_sps_normalized.append((coded_sp - coded_sps_mean) / coded_sps_std)
    return coded_sps_normalized, coded_sps_mean, coded_sps_std

def coded_sp_statistics(coded_sps):
    # sp shape (T, D)
    coded_sps_concatenated = np.concatenate(coded_sps, axis = 0)
    coded_sps_mean = np.mean(coded_sps_concatenated, axis = 0, keepdims = False)
    coded_sps_std = np.std(coded_sps_concatenated, axis = 0, keepdims = False)
    return coded_sps_mean, coded_sps_std

def normalize_coded_sp(coded_sp, coded_sp_mean, coded_sp_std):
    normed = (coded_sp - coded_sp_mean) / coded_sp_std
    return normed

def coded_sps_normalization_transoform(coded_sps, coded_sps_mean, coded_sps_std):

    coded_sps_normalized = list()
    for coded_sp in coded_sps:
        coded_sps_normalized.append((coded_sp - coded_sps_mean) / coded_sps_std)
    
    return coded_sps_normalized

def coded_sps_normalization_inverse_transoform(normalized_coded_sps, coded_sps_mean, coded_sps_std):

    coded_sps = list()
    for normalized_coded_sp in normalized_coded_sps:
        coded_sps.append(normalized_coded_sp * coded_sps_std + coded_sps_mean)

    return coded_sps

def coded_sp_padding(coded_sp, multiple = 4):
    num_features = coded_sp.shape[0]
    num_frames = coded_sp.shape[1]
    num_frames_padded = int(np.ceil(num_frames / multiple)) * multiple
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    coded_sp_padded = np.pad(coded_sp, ((0, 0), (num_pad_left, num_pad_right)), 'constant', constant_values = 0)
    return coded_sp_padded

def wav_padding(wav, sr, frame_period, multiple = 4):

    assert wav.ndim == 1 
    num_frames = len(wav)
    num_frames_padded = int((np.ceil((np.floor(num_frames / (sr * frame_period / 1000)) + 1) / multiple + 1) * multiple - 1) * (sr * frame_period / 1000))
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    wav_padded = np.pad(wav, (num_pad_left, num_pad_right), 'constant', constant_values = 0)

    return wav_padded

def logf0_statistics(f0s):
    log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
    log_f0s_mean = log_f0s_concatenated.mean()
    log_f0s_std = log_f0s_concatenated.std()

    return log_f0s_mean, log_f0s_std

def pitch_conversion(f0, mean_log_src, std_log_src, mean_log_target, std_log_target):

    # Logarithm Gaussian normalization for Pitch Conversions
    f0_converted = np.exp((np.ma.log(f0) - mean_log_src) / std_log_src * std_log_target + mean_log_target)

    return f0_converted

def wavs_to_specs(wavs, n_fft = 1024, hop_length = None):

    stfts = list()
    for wav in wavs:
        stft = librosa.stft(wav, n_fft = n_fft, hop_length = hop_length)
        stfts.append(stft)

    return stfts


def wavs_to_mfccs(wavs, sr, n_fft = 1024, hop_length = None, n_mels = 128, n_mfcc = 24):

    mfccs = list()
    for wav in wavs:
        mfcc = librosa.feature.mfcc(y = wav, sr = sr, n_fft = n_fft, hop_length = hop_length, n_mels = n_mels, n_mfcc = n_mfcc)
        mfccs.append(mfcc)

    return mfccs


def mfccs_normalization(mfccs):

    mfccs_concatenated = np.concatenate(mfccs, axis = 1)
    mfccs_mean = np.mean(mfccs_concatenated, axis = 1, keepdims = True)
    mfccs_std = np.std(mfccs_concatenated, axis = 1, keepdims = True)

    mfccs_normalized = list()
    for mfcc in mfccs:
        mfccs_normalized.append((mfcc - mfccs_mean) / mfccs_std)
    
    return mfccs_normalized, mfccs_mean, mfccs_std


def sample_train_data(dataset_A, dataset_B, n_frames = 128):

    num_samples = min(len(dataset_A), len(dataset_B))
    train_data_A_idx = np.arange(len(dataset_A))
    train_data_B_idx = np.arange(len(dataset_B))
    np.random.shuffle(train_data_A_idx)
    np.random.shuffle(train_data_B_idx)
    train_data_A_idx_subset = train_data_A_idx[:num_samples]
    train_data_B_idx_subset = train_data_B_idx[:num_samples]

    train_data_A = list()
    train_data_B = list()

    for idx_A, idx_B in zip(train_data_A_idx_subset, train_data_B_idx_subset):
        data_A = dataset_A[idx_A]
        frames_A_total = data_A.shape[1]
        assert frames_A_total >= n_frames
        start_A = np.random.randint(frames_A_total - n_frames + 1)
        end_A = start_A + n_frames
        train_data_A.append(data_A[:,start_A:end_A])

        data_B = dataset_B[idx_B]
        frames_B_total = data_B.shape[1]
        assert frames_B_total >= n_frames
        start_B = np.random.randint(frames_B_total - n_frames + 1)
        end_B = start_B + n_frames
        train_data_B.append(data_B[:,start_B:end_B])

    train_data_A = np.array(train_data_A)
    train_data_B = np.array(train_data_B)

    return train_data_A, train_data_B