# AI Music Assistant  
_Intelligent Melody Continuation Using Deep-Learning & Classical Methods_

This repository accompanies my UCL BSc Final-Year Project.  
It contains the code and lightweight assets needed to **train, evaluate and serve** four melody-continuation models (Markov, custom LSTM, MelodyRNN, MusicVAE) plus two Streamlit front-ends:

* **`ai-assistant-app.py`** – interactive assistant for composers  
* **`user-testing-app.py`** – blind-listening study used in the dissertation

---

## 1  Folder structure (tracked in Git)

| Path | Purpose |
|------|---------|
| `.devcontainer/` | VS Code remote-container setup |
| **applications/** | Utility scripts & notebooks used during development (e.g. MIDI→video) |
| **magenta/** | Thin wrappers around Google Magenta models & MIDI helpers |
| **models/** | All custom training code, checkpoints and generated MIDI <br> ├─ **lstm/** … custom LSTM <br> └─ **markov/** … Markov baseline |
| **sounds/FluidR3_GM.sf2** | Default GM sound-font for quick audio preview |
| `requirements.txt` | Python dependencies (works with ≥3.9) |
| `packages.txt` | System packages required by Ubuntu (read by the DevContainer) |
| `testing.py` | Regression tests |

---

## 2  Large assets **not** pushed to Git (LFS/Git-ignored)

To keep the repo <100 MB and respect upstream licences, several bulky or external-licence folders are **ignored**.  
Follow the table to reproduce the full local tree expected by the scripts:

| Local folder (ignored) | Why it’s needed | How to obtain |
|------------------------|-----------------|---------------|
| **`magenta/magenta/`** | Google Magenta source (for MelodyRNN & MusicVAE) | ```bash<br># from repo root<br>git clone --depth 1 https://github.com/magenta/magenta.git magenta/magenta<br>pip install -e magenta/magenta[extras]<br>``` |
| **`magenta/magenta_setup/`** | Convenience scripts from the same repo | Already included when you clone above |
| **`checkpoints/`** | Pre-trained Magenta bundles & VAEs | ```bash<br>mkdir -p checkpoints && cd checkpoints<br># MelodyRNN bundles (≈ 20 MB each)<br>wget https://storage.googleapis.com/magentadata/models/melody_rnn/attention_rnn.mag<br>wget https://storage.googleapis.com/magentadata/models/melody_rnn/basic_rnn.mag<br>wget https://storage.googleapis.com/magentadata/models/melody_rnn/lookback_rnn.mag<br>wget https://storage.googleapis.com/magentadata/models/melody_rnn/mono_rnn.mag<br><br># 16-bar hierarchical MusicVAE (≈ 50 MB)<br>wget https://storage.googleapis.com/magentadata/models/music_vae/checkpoints/hierdec-mel_16bar.tar<br>tar -xf hierdec-mel_16bar.tar && rm hierdec-mel_16bar.tar<br>``` |
| **`Fluid_R3_GM/`** & **`fluidsynth-2.4.0/`** | Optional local build of FluidSynth & large GM sound-font | Use your package manager instead:<br>`sudo apt install fluidsynth fluid-soundfont-gm`<br>or<br>`brew install fluid-synth` |
| **`models/*/pickles/`** | Markov & LSTM training pickles built from the _Mono-MIDI Transposition Dataset_ | ```bash<br>python models/markov/train_markov.py --download_dataset<br>python models/lstm/train_lstm.py   --download_dataset<br>``` (Scripts download and cache the dataset automatically) |
| **`models/lstm/old-dataset/13369389/`** | Folk tune dataset from Zenodo used in early experiments | ```bash<br>wget https://zenodo.org/record/13369389/files/13369389.zip -O folk.zip<br>unzip folk.zip -d models/lstm/old-dataset/13369389 && rm folk.zip<br>``` |

> **Tip:** run `git lfs install` first if you decide to track any of these large files yourself.

---

## 3  Quick start

### 3.1 Clone & set-up

```bash
git clone https://github.com/<your-fork>/ai-music-assistant.git
cd ai-music-assistant

# Python env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Install system deps (Ubuntu)
xargs sudo apt-get install -y < packages.txt
```

Then fetch the _ignored_ assets from section 2.

### 3.2 Launch the apps

| Action | Command |
|--------|---------|
| Interactive assistant | `streamlit run ai-assistant-app.py` |
| Blind listening study | `streamlit run user-testing-app.py` |

Both apps default to the **sounds/FluidR3_GM.sf2** sound-font and write outputs to `applications/output_videos` or `models/*/output_midis`.

### 3.3 Train from scratch (optional)

```bash
# Markov baseline
python models/markov/train_markov.py

# Custom LSTM (GPU recommended)
python models/lstm/train_lstm.py --epochs 10
```

The scripts rebuild the pickles/datasets if absent.

### 3.4 Generate a continuation from CLI

```bash
python applications/models/generate_lstm_continuation.py \
    --primer_path input.mid \
    --output_path continuation.mid \
    --model_path models/lstm/lstm_model.h5
```

---

## 4  Reproducing the dissertation results

1. Run `user-testing-app.py` locally or deploy to Streamlit Cloud.  
2. Collect responses (stored in a Supabase table; see `supabase/schema.sql`).  
3. Analyse ratings with `testing.py`, which recreates all plots in the report.

---

## 5  Project citation

If you build on this work please cite:

```
@undergraduate{Gildea2025,
  title   = {AI Music Assistant: Intelligent Music Prediction Using Deep Learning},
  author  = {Gildea, N.},
  school  = {University College London},
  year    = {2025}
}
```

---

## 6  Licence & acknowledgements

* **Code** – MIT Licence (see `LICENSE`).  
* **Datasets** – Licences as per their respective repositories:  
  * _Mono-MIDI Transposition Dataset_ © Sebastián G. Verde – CC BY 4.0  
  * _Folk tune dataset_ (Zenodo 13369389) – Creative Commons  
* **Magenta** and checkpoints © Google – Apache 2.0  
* **FluidR3_GM.sf2** – CC0

> This project would not be possible without Google Magenta, Fluidsynth, Streamlit and the open-source music-AI community – thank you!

---

Created with ❤️ for my UCL BSc final-year project.
