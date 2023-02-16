import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from dataset import Dataset, amp_pha_spectra, get_dataset_filelist
from models import NSPP_Model, losses
from utils import AttrDict, build_env, scan_checkpoint, load_checkpoint, save_checkpoint, save_info

torch.backends.cudnn.benchmark = True


def train(h):

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:0')

    NSPP = NSPP_Model(h).to(device)

    print(NSPP)
    os.makedirs(h.checkpoint_path, exist_ok=True)
    print("checkpoints directory : ", h.checkpoint_path)

    if os.path.isdir(h.checkpoint_path):
        cp = scan_checkpoint(h.checkpoint_path, 'NSPP_')
        info = scan_checkpoint(h.checkpoint_path, 'info_')

    steps = 0
    if cp is None:
        state_dict_info = None
        last_epoch = -1
    else:
        state_dict = load_checkpoint(cp, device)
        state_dict_info = load_checkpoint(info, device)
        NSPP.load_state_dict(state_dict['model'])
        steps = state_dict_info['steps'] + 1
        last_epoch = state_dict_info['epoch']

    optim = torch.optim.AdamW(NSPP.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_info is not None:
        optim.load_state_dict(state_dict_info['optim'])

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(h.input_training_wav_list, h.input_validation_wav_list)

    trainset = Dataset(training_filelist, h.segment_size, h.n_fft,
                       h.hop_size, h.win_size, h.sampling_rate, split=True, shuffle=True, n_cache_reuse=0, device=device)

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=None,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)

    validset = Dataset(validation_filelist, h.segment_size, h.n_fft,
                       h.hop_size, h.win_size, h.sampling_rate, split=False, shuffle=False, n_cache_reuse=0, device=device)
    
    validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                   sampler=None,
                                   batch_size=1,
                                   pin_memory=True,
                                   drop_last=True)

    sw = SummaryWriter(os.path.join(h.checkpoint_path, 'logs'))

    NSPP.train()

    for epoch in range(max(0, last_epoch), h.training_epochs):

        start = time.time()
        print("Epoch: {}".format(epoch+1))

        for i, batch in enumerate(train_loader):
            start_b = time.time()
            log_amplitude, phase = batch
            log_amplitude = torch.autograd.Variable(log_amplitude.to(device, non_blocking=True))
            phase = torch.autograd.Variable(phase.to(device, non_blocking=True))

            phase_g = NSPP(log_amplitude)

            optim.zero_grad()

            L_IP, L_GD, L_IAF = losses(phase, phase_g, h.n_fft, phase.size()[-1])

            loss_all = L_IP + L_GD + L_IAF

            loss_all.backward()
            optim.step()

            if steps % h.stdout_interval == 0:
                print('Steps : {:d}, Total Loss: {:4.3f}, Instantaneous Phase Loss : {:4.3f}, Group Delay Loss : {:4.3f}, Instantaneous Angular Frequency Loss : {:4.3f}, s/b : {:4.3f}'.
                      format(steps, loss_all, L_IP, L_GD, L_IAF, time.time() - start_b))

            if steps % h.checkpoint_interval == 0 and steps != 0:
                checkpoint_path = "{}/NSPP_{:08d}".format(h.checkpoint_path, steps)
                info_path = "{}/info_{:08d}".format(h.checkpoint_path, steps)
                save_checkpoint(checkpoint_path,
                                {'model': NSPP.state_dict()})
                save_info(info_path,
                          {'optim': optim.state_dict(),
                           'steps': steps,
                           'epoch': epoch})

            if steps % h.summary_interval == 0:
                sw.add_scalar("Training/Total Loss", loss_all, steps)

            if steps % h.validation_interval == 0 and steps != 0:
                NSPP.eval()
                torch.cuda.empty_cache()
                val_L_IP_total = 0
                val_L_GD_total = 0
                val_L_IAF_total = 0
                with torch.no_grad():
                    for j, batch in enumerate(validation_loader):
                        log_amplitude, phase = batch
                        phase_g = NSPP(log_amplitude.to(device))
                        phase = torch.autograd.Variable(phase.to(device, non_blocking=True))

                        val_L_IP, val_L_GD, val_L_IAF = losses(phase, phase_g, h.n_fft, phase.size()[-1])
                        val_L_IP_total += val_L_IP.item()
                        val_L_GD_total += val_L_GD.item()
                        val_L_IAF_total += val_L_IAF.item()

                    val_L_IP = val_L_IP_total / (j+1)
                    val_L_GD = val_L_GD_total / (j+1)
                    val_L_IAF = val_L_IAF_total / (j+1)

                    sw.add_scalar("Validation/Total Loss", val_L_IP + val_L_GD + val_L_IAF, steps)
                    sw.add_scalar("Validation/Instantaneous Phase Loss", val_L_IP, steps)
                    sw.add_scalar("Validation/Group Delay Loss", val_L_GD, steps)
                    sw.add_scalar("Validation/Instantaneous Angular Frequency Loss", val_L_IAF, steps)

                NSPP.train()

            steps += 1

        scheduler.step()
        
        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('NSPP Training..')

    config_file = 'config.json'

    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(config_file, 'config.json', h.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
    else:
        pass

    train(h)


if __name__ == '__main__':
    main()
