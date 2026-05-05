# BioDCASE 2026 TinyML Bird Sound Dataset

## Overview
This dataset contains annotated avian acoustic recordings tailored for the [BioDCASE 2026 TinyML Task 3 challenge](https://biodcase.github.io/challenge2026/task3). It is designed to facilitate the development and evaluation of lightweight machine learning models capable of identifying bird species from audio recordings in resource-constrained environments.

The dataset includes recordings of several common European bird species as well as a dedicated background noise class representing urban/town environments. Original recordings were made at various sites in Germany using eciPi recording units by OekoFor, and all samples have been carefully annotated by Ralph Martin to ensure high-quality call centers.

## Technical Specifications
- **Format:** `.wav`
- **Sample Rate:** 24,000 Hz
- **Channels:** 1 (Mono)
- **Bit Depth:** 16-bit PCM
- **Duration:** 3.0 seconds per snippet
- **Total Runtime (Development Set):** ~2.29 hours

## Categories
The dataset consists of **10 bird species** plus **1 background class** (11 classes in total).

*(Note: The exact list of 10 species depends on the specific source selection but generally includes typical garden/forest birds such as Common Chaffinch, Eurasian Blackbird, Great Tit, etc., as well as a 'Background' class).*

## Dataset Structure
The dataset provided here is the Development Set (for training and validation). A separate, balanced Test Set is held out and kept hidden for the final challenge evaluation. The distribution per class in the development set is:

- **Development Set (`BioDCASE2026_TinyML_Development_Dataset.zip`)**
  - **Train:** 200 samples per class
  - **Validation:** 50 samples per class
  *(Note: A hidden/hold-out test set containing 50 samples per class is maintained separately for scoring challenge submissions).*

Each split contains subfolders corresponding to the species/class name.

### Folder Structure Example
```
Development_Set/
├── Train/
│   ├── Background/
│   ├── Common_Chaffinch/
│   ├── ...
├── Validation/
│   ├── Background/
│   ├── Common_Chaffinch/
│   ├── ...
```

## File Naming Convention
Files are systematically named to easily identify their split and ground-truth species:
`BioDCASE2026_TinyML_[SPLIT]_[ID]_[Species].wav`

- `[SPLIT]`: TRAIN, VAL, or TEST
- `[ID]`: A 4-digit sequential identifier (e.g., 0001)
- `[Species]`: The formatted species name (e.g., Common_Chaffinch)

**Example:** `BioDCASE2026_TinyML_TRAIN_0042_Common_Chaffinch.wav`

## License
This dataset is published under the Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) license.

