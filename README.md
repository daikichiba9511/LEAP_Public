# LEAP - Atmospheric Physics using AI (ClimSim)

this repository is my 55th🥉 solution of LEAP - Atmospheric Physics using AI (ClimSim) on Kaggle.

## Overview

- Sub1:
  - exp089
  - Public: 0.75854, Private: 0.75402
  - ensemble: exp078, exp080, exp081, exp082, exp084, exp087
- Sub2:
  - exp090
  - Public: 0.75933, Private: 0.75461
  - ensemble: exp081, exp082, exp084, exp087

- EXP Log Overview
  - exp078:
    - SE Residual Unet1D (64, 128, 256, 512, 1024) + MSE loss + seed 1995 + sample_size 8_000_000
    - CV: 0.701370534087066
  - exp080:
    - SE Residual Unet1D (64, 128, 256, 512, 1024) + TransformerEncoder as MiddleBlock + MSE loss + seed 1995 + sample_size 8_000_000
    - CV: 0.6984481559367275
  - exp081:
    - SE Residual Unet1D (128, 256, 512, 1024, 2048) + LSTM + MSE loss + seed 42 + sample_size 6_000_000
    - CV: 0.70824509367178
  - exp082:
    - SE Residual Unet1D (128, 256, 512, 1024, 2048) + LSTM + MSE loss + seed 2024 + sample_size 8_000_000
    - CV: 0.7142930243451985
  - exp084:
    - SE Residual Unet1D (128, 256, 512, 1024, 2048) + LSTM + huver loss + seed 1995 + sample_size 8_000_000
    - CV: 0.7086811296875283
  - exp087:
    - SE Residual Unet1D (128, 256, 512, 1024, 2048) + LSTM + (0.5 *MSE loss + 0.5* huver loss) + seed 1994 + sample_size 8_000_000
    - CV: 0.715509

- Unet base Architecture
  - 1D Conv
  - AdaptiveAvgPool1d
  - SE Layer
  - Skip connect
  - LSTM Neck
  - CNN Head

- Sample size 8_000_000 -> sampling {7.25, 7.3} * 10**7

- Model Overview
  - <https://x.com/nnc_5522/status/1818119455467331816>

## Usage

1. download the competition data & unzip it to `input/`
2. `docker compose up -d && docker compose exec local-dev bash`
3. `make setup`
4. `python -m scripts.random_sampling_train`
5. `python -m run`
6. `python -m src.exp.exp089.ensemble && python -m src.exp.exp090.ensemlbe`

