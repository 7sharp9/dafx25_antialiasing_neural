import torch
import numpy as np

def get_config(idx: int):

    # base config
    conf = {
        'sample_rate': 44100,
        'train_data': {
            'midi_min': 21,
            'dB_min': -60,
            'dB_max': 0,
            'num_tones': 1600,
            'dur': 1.2,
            'linear_f0_sample': False
        },
        'val_data': {
            'dB_min': -20,
            'midi_min': 21,
            'randomise': False,
            'num_tones': 88,
            'dur': 1.2,
        },
        'audio_val_data': {
            'dur': 4,
            'dB': 0,
        },

        'batch_size': {
            'train': 40 if torch.cuda.is_available() else 16,
            'val': 16 if torch.cuda.is_available() else 8,
            'audio_val': 4
        },

        'loss_weights': {
            'mesr': 0.0,
            'esr': 0.0,
            'asr': 0.0,
            'esr_normal': 0.5,
            'nmr': 0.5,
            'dc': 1.0,
        },

        'pre_emph': 'lp',
        'lr': 5e-4,
        'max_epochs': 100,
        'tbptt_steps': 4410,

        'training': {
            'use_ema': True,
            'ema_decay': 0.999,
            'gradient_clip_norm': 1.0,
            'use_amp': False,  # Keep double precision per paper
        },

        'scheduler': {
            'type': 'cosine_warm_restarts',
            'T_0': 10,
            'T_mult': 2,
            'eta_min_ratio': 0.01,
        }
    }

    # Proteus models --------------------
    if idx == 0:
        conf['model_json'] = 'weights/Proteus_Tone_Packs/AmpPack1/MesaMiniRec_HighGain_DirectOut.json'
        conf['model_name'] = 'MesaMiniRec'

    if idx == 1:
        conf['model_json'] = 'weights/Proteus_Tone_Packs/PedalPack1/GoatPedal_HighGain.json'
        conf['model_name'] = 'Goat'

    # NAM models -------------------------
    elif idx == 2 or idx == 3:

        conf['sample_rate'] = 48000
        conf['tbptt_steps'] = 24000
        conf['batch_size']['train'] = 40 if torch.cuda.is_available() else 16
        conf['train_data']['dur'] = 1.05
        conf['lr'] = 5e-4

        if idx == 2:
            conf['model_json'] = 'weights/NAM/Vox AC15/Vox AC15CH Overdriven Normal.nam'
            conf['model_name'] = 'VoxAC15_OD_Normal'

        if idx == 3:
            conf['model_json'] = 'weights/NAM/Marshall JCM 800 2203/JCM800 2203 - P5 B5 M5 T5 MV7 G10 - AZG - 700.nam'
            conf['model_name'] = 'JCM800'
    # ---------------------------------------

    # Custom LSTM models--------------------------------
    if idx == 4:
        conf['model_json'] = 'weights/pre-trained-by-us/broadcast_lstm_teacher.json'
        conf['model_name'] = 'BroadcastLSTM'
    if idx == 5:
        conf['model_json'] = 'weights/pre-trained-by-us/jhm8_lstm_teacher.json'
        conf['model_name'] = 'GypsyLSTM'

    # Custom TCN models__________________________________
    if idx == 6 or idx == 7:
        conf['tbptt_steps'] = 22050
        conf['batch_size']['train'] = 32 if torch.cuda.is_available() else 16
        conf['train_data']['dur'] = 1.05
        conf['lr'] = 5e-4
        if idx == 6:
            conf[
                'model_json'] = 'weights/pre-trained-by-us/broadcast_tcn_teacher.nam'
            conf['model_name'] = 'BroadcastTCN'
        if idx == 7:
            conf[
                'model_json'] = 'weights/pre-trained-by-us/jhm8_tcn_teacher.nam'
            conf['model_name'] = 'GypsyTCN'
    # ________________________________________________

    return conf
