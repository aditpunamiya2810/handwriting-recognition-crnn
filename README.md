# Bilingual Handwriting Recognition (English + Hindi) using CRNN

A deep learning project that recognizes handwritten **English and Hindi (Devanagari)** words using a **CRNN (CNN + BiLSTM) trained with CTC loss**, built in PyTorch.

## Overview
Handwritten text varies a lot between writers, and Devanagari adds complexity with its conjuncts and diacritics. This project trains a single model to read both English and Hindi words from images without needing character-level segmentation.

## Features
- Single model for English and Hindi
- CNN feature extractor + 2-layer Bidirectional LSTM
- CTC loss for alignment-free sequence prediction
- Data augmentation (affine, shear, brightness/contrast jitter)
- Greedy CTC decoding
- Visual testing script showing true vs. predicted text

## Model Architecture
| Stage | Details |
|-------|---------|
| Input | Grayscale image, 64 x 400 |
| CNN | 2 x (Conv 3x3 + ReLU + MaxPool 2x2), 32 → 64 filters |
| Bridge | Linear layer (feature maps → 64) |
| RNN | 2-layer BiLSTM, 128 hidden units, dropout 0.25 |
| Output | Linear + LogSoftmax over vocabulary + CTC blank |

## Dataset
The dataset was **created from scratch**: handwritten samples of [200] words ([100] English and [100] Hindi) written by [N] different writers/styles. Each word was written, scanned/photographed, and labeled manually.

> The dataset is not included in this repository.

Expected structure if you want to build your own:
```
handwriting_dataset/
├── style_1/
│   ├── english/   (1.png, 2.png, ...)
│   └── hindi/     (1.png, 2.png, ...)
├── style_2/
...
```
Images are numbered in the same order as the word lists in `prepare_dataset.py`.

## Training Details
| Parameter | Value |
|-----------|-------|
| Epochs | 200 |
| Batch size | 32 |
| Optimizer | Adam (lr = 0.001) |
| Loss | CTC Loss |
| Image size | 64 x 400 |
| Train/Val split | 90% / 10% |

## Results
| Metric | Value |
|--------|-------|
| Final training loss | [x.xxxx] |
| Character accuracy | [xx%] |
| Word accuracy | [xx%] |

Sample predictions:

| True | Predicted |
|------|-----------|
| [Water] | [Water] |
| [पानी] | [पानी] |

(You can add screenshots from `testing.ipynb` here.)

## Getting Started

### Installation
```bash
git clone https://github.com/aditpunamiya2810/handwriting-recognition-crnn.git
cd handwriting-recognition-crnn
pip install -r requirements.txt
```

### Usage
1. Prepare your dataset in the structure above.
2. Generate labels and the vocabulary:
```bash
   python prepare_dataset.py
```
3. Train:
```bash
   python train.py
```
4. Test with the pretrained model:
```bash
   python test.py
```

Pretrained weights are in `models/`. `_v2` was trained with data augmentation.

## Project Structure
- `prepare_dataset.py`: builds `labels.csv` and `characters.txt`
- `train.py`: model, dataset, and training loop
- `test.py`: loads the model and visualizes predictions
- `notebooks/testing.ipynb`: testing experiments
- `docs/`: project report and training logs

## Limitations and Future Work
- Limited vocabulary (word-level samples only)
- Small number of writers
- Add beam search / language-model decoding
- Extend to full sentences and more writers

## Team
- [Name 1] (I003)
- [Name 2] (I025)
- [Name 3] (I048)

## License
MIT License
