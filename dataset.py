import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
import librosa

def load_wav(full_path, sample_rate):
    data, _ = librosa.load(full_path, sr=sample_rate, mono=True)
    return data

def amp_pha_spectra(y, n_fft, hop_size, win_size):

	#y:[1, T] or [batch_size, T]
	
	hann_window=torch.hann_window(win_size).to(y.device)

	stft_spec=torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,center=True)

	rea=stft_spec[:,:,:,0] #[batch_size, n_fft//2+1, frames]
	imag=stft_spec[:,:,:,1] #[batch_size, n_fft//2+1, frames]

	log_amplitude=torch.log(torch.abs(torch.sqrt(torch.pow(rea,2)+torch.pow(imag,2)))+1e-5) #[batch_size, n_fft//2+1, frames]
	phase=torch.atan2(imag,rea) #[batch_size, n_fft//2+1, frames]

	return log_amplitude, phase

def get_dataset_filelist(input_training_wav_list,input_validation_wav_list):

    with open(input_training_wav_list, 'r') as fi:
        training_files = [x for x in fi.read().split('\n') if len(x) > 0]

    with open(input_validation_wav_list, 'r') as fi:
        validation_files = [x for x in fi.read().split('\n') if len(x) > 0]

    return training_files, validation_files


class Dataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft,
                 hop_size, win_size, sampling_rate, split=True, shuffle=True, n_cache_reuse=1,device=None):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device

    def __getitem__(self, index):
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            audio=load_wav(filename, self.sampling_rate)
            self.cached_wav = audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio) #[T]
        audio = audio.unsqueeze(0) #[1,T]

        if self.split:
            if audio.size(1) >= self.segment_size:
                max_audio_start = audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                audio = audio[:, audio_start:audio_start+self.segment_size] #[1,T]
            else:
                audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        log_amplitude, phase = amp_pha_spectra(audio, self.n_fft, self.hop_size, self.win_size) #[1,n_fft/2+1,frames]

        return (log_amplitude.squeeze(), phase.squeeze())

    def __len__(self):
        return len(self.audio_files)
