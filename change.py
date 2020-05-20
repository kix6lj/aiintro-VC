import librosa
import numpy as np
import os
import pyworld
from pycwt import wavelet
import pywt
from sklearn import preprocessing
from scipy.interpolate import interp1d
from scipy.signal import firwin, lfilter

def world_decompose(wav, fs = 16000 , frame_period = 5.0):
    # Decompose speech signal into f0, spectral envelope and aperiodicity using WORLD
    wav = wav.astype(np.float64)
    f0, timeaxis = pyworld.harvest(wav, fs, frame_period = frame_period, f0_floor = 71.0, f0_ceil = 800.0)
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs)
    ap = pyworld.d4c(wav, f0, timeaxis, fs)
    return f0, timeaxis, sp, ap

def world_decode_spectral_envelop(coded_sp, fs):
    # Decode Mel-cepstral to sp
    fftlen = pyworld.get_cheaptrick_fft_size(fs)
    decoded_sp = pyworld.decode_spectral_envelope(coded_sp, fs, fftlen)
    return decoded_sp

def world_speech_synthesis(f0, coded_sp, ap, fs = 16000 , frame_period = 5.0):
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

def Change(wav,style):
    f0, timeaxis, sp, ap = world_decompose(wav)
    f0_converted, timeaxis, coded_sp_converted, ap = Converter.convert(f0, timeaxis, sp, ap, style)
    wav_converted = world_speech_synthesis(f0_converted , coded_sp_converted, ap)
    return wav_converted

