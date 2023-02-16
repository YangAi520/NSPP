from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from utils import AttrDict
from dataset import load_wav, amp_pha_spectra
from models import NSPP_Model
import soundfile as sf
import librosa
import numpy as np

h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def generation(h):
    NSPP = NSPP_Model(h).to(device)

    state_dict = load_checkpoint(h.checkpoint_file_load, device)
    NSPP.load_state_dict(state_dict['model'])

    filelist = sorted(os.listdir(h.test_input_log_amp_dir if h.test_log_amp_load else h.test_input_wav_dir))

    os.makedirs(h.test_output_dir, exist_ok=True)

    NSPP.eval()

    with torch.no_grad():
        for i, filename in enumerate(filelist):

            if h.test_log_amp_load:
                log_amplitude = np.load(os.path.join(h.test_input_log_amp_dir, filename))
                log_amplitude = torch.FloatTensor(log_amplitude).unsqueeze(0).to(device)
            else:
                raw_wav, _ = librosa.load(os.path.join(h.test_input_wav_dir, filename), sr=h.sampling_rate, mono=True)
                raw_wav = torch.FloatTensor(raw_wav).to(device)
                log_amplitude, _ = amp_pha_spectra(raw_wav.unsqueeze(0), h.n_fft, h.hop_size, h.win_size)

            phase_g = NSPP(log_amplitude)

            real_part = torch.exp(log_amplitude)* torch.cos(phase_g)
            imaginary_part = torch.exp(log_amplitude)* torch.sin(phase_g)

            stft_spec = torch.cat((real_part.unsqueeze(-1), imaginary_part.unsqueeze(-1)),-1)

            audio_g = torch.istft(stft_spec, h.n_fft, hop_length=h.hop_size, win_length=h.win_size, window=torch.hann_window(h.win_size).to(device), center=True)

            phase_g = phase_g.squeeze()
            audio_g = audio_g.squeeze()

            phase_g = phase_g.cpu().numpy()
            audio_g = audio_g.cpu().numpy()

            np.save(os.path.join(h.test_output_dir, filename.split('.')[0]+'_phase.npy'), phase_g)
            sf.write(os.path.join(h.test_output_dir, filename.split('.')[0]+'.wav'), audio_g, h.sampling_rate, 'PCM_16')

            print(filename.split('.')[0])


def main():
    print('NSPP Generation..')

    config_file = 'config.json'

    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    generation(h)

if __name__ == '__main__':
    main()

