<h2 style="font-size: 1.5em" align="center">Anti-aliasing of neural distortion effects via  model fine tuning</h2>
<p style="font-size: 1.0em" align="center">
Alistair Carson, Alec Wright and Stefan Bilbao
</p>
<p style="font-size: 0.75em" align="center">
<i><a href="https://www.acoustics.ed.ac.uk/" target="_blank" rel="noopener noreferrer">Acoustics and Audio Group</a><br>University of Edinburgh</i> <br>Edinburgh, UK
</p>
<p style="font-size: 1.0em; text-align: center">
Accompanying code for the DAFx25 paper.</p>
<div style="text-align: center">
    <a href="https://a-carson.github.io/dafx25_antialiasing_neural/" 
        class="btn btn--primary btn--small"
        target="_blank" rel="noopener noreferrer">
    🔊 Audio Examples
    </a> &ensp; &ensp; &ensp; &ensp;  <a href="https://arxiv.org/abs/2505.11375" 
        class="btn btn--primary btn--small"
        target="_blank" rel="noopener noreferrer">
      🗞️ Paper
    </a>
</div>


#### Requirements

First, clone this repo and the initialise the OpenAmp submodule:
```angular2html
git clone --recurse-submodules git@github.com:a-carson/dafx25_antialiasing_neural.git
```

Create conda environment:
```
conda env create -f env.yaml
conda activate aa_neural
```

(optional) Download Proteus Tone Packs collection of pre-trained models:

```
curl -LO https://github.com/GuitarML/ToneLibrary/releases/download/v1.0/Proteus_Tone_Packs.zip
tar -xf Proteus_Tone_Packs.zip
rm Proteus_Tone_Packs.zip
mv Proteus_Tone_Packs weights/
```
(optional) Download the Neural Amp Modeler (NAM) weights (TCN models) which we used from the links below, and extract into `weights/` directory:

- Marshall JCM: https://www.tone3000.com/tones/marshall-jcm-800-2203-1071

- Vox AC15: https://www.tone3000.com/tones/vox-ac15-1080

#### Run training
Run fine tuning anti-aliasing training:
```
python3 train.py --config <config_idx>
```
where `config_idx` can be chosen from the following list (default: 0)
```angular2html
config_idx      Target Pedal/Amp        Model          Initial weights dir
---------------------------------------------------------------------------------------------------
0               Mesa Mini Rectifier     LSTM           weights/Proteus_Tone_Packs/AmpPack1/MesaMiniRec_HighGain_DirectOut.json
1               Goat Pedal              LSTM           weights/Proteus_Tone_Packs/PedalPack1/GoatPedal_HighGain.json
2               Vox AC15                TCN            weights/NAM/Vox AC15/Vox AC15CH Overdriven Normal.nam
3               Marshall JCM            TCN            weights/NAM/Marshall JCM 800 2203/JCM800 2203 - P5 B5 M5 T5 MV7 G10 - AZG - 700.nam
4               Hudson Broadcast        LSTM           weights/pre-trained-by-us/broadcast_lstm_teacher.json
5               JHM8 Fuzz               LSTM           weights/pre-trained-by-us/jhm8_lstm_teacher.json 
6               Hudson Broadcast        TCN            weights/pre-trained-by-us/broadcast_tcn_teacher.nam
7               JHM8 Fuzz               TCN            weights/pre-trained-by-us/jhm8_tcn_teacher.nam
```

#### Google Colab Training (GPU Accelerated)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/7sharp9/dafx25_antialiasing_neural/blob/main/colab-run.ipynb)

For faster training with GPU acceleration, use the provided Colab notebook.

**Setup:**
1. Create `AA_Neural/` folder in your Google Drive root
2. Upload `weights/NAM/Marshall JCM 800 2203/` directory to `Drive/AA_Neural/weights/NAM/`
3. Upload `audio_data/val_input.wav` to `Drive/AA_Neural/audio_data/`
4. Click the badge above to open the notebook in Colab
5. Update `REPO_URL` in the notebook (cell 2) with your fork or this repo URL
6. Run all cells (Runtime → Run all)

**Performance comparison:**
- Local CPU: ~32 min/epoch
- Colab T4 GPU: ~1-2 min/epoch (20-30x faster)

See `colab-run.ipynb` for detailed instructions and troubleshooting.

#### Citation
If using this work please use the following citation:
```angular2html
@inproceedings{Carson2025_antialiasing,
    author = {Carson, A. and Wright, A. and Bilbao, S.},
    title = {Anti-aliasing of neural distortion effects via model fine-tuning},
    booktitle = {Proceedings of the 28th International Conference on Digital Audio Effects (DAFx25)},
    year = {2025},
    month = {Sept.},
    address = {Ancona, Italy}
}
```
