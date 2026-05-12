# Xeno-canto Dataset Builder

This folder contains tooling to download external Xeno-canto data for the 10 BioDCASE 2026 target bird species.

This is an **external-data track**. Do not mix these files with the no-external-data baseline unless the submission is explicitly declared as using Xeno-canto. The BioDCASE rules require external datasets/trained models to be public, freely accessible, available before 2025-04-01, clearly referenced, and communicated to organizers where allowed.

## Target Species

The `species.json` file maps the 10 BioDCASE bird classes to scientific names:

- Common Chaffinch: `Fringilla coelebs`
- Common Chiffchaff: `Phylloscopus collybita`
- Eurasian Blackbird: `Turdus merula`
- Eurasian Blackcap: `Sylvia atricapilla`
- Eurasian Blue Tit: `Cyanistes caeruleus`
- Great Spotted Woodpecker: `Dendrocopos major`
- Great Tit: `Parus major`
- Mallard: `Anas platyrhynchos`
- Song Thrush: `Turdus philomelos`
- Tawny Owl: `Strix aluco`

`Background` is not downloaded from Xeno-canto. In BioDCASE it is a real 11th class representing non-target/background sound, not a bird species.

## API Key

Xeno-canto API v3 requires an API key. Create one from your Xeno-canto account page and export it locally:

```bash
export XC_API_KEY="your_key_here"
```

Never commit the key.

## Metadata-Only Test

Run from this folder:

```bash
cd biodcase_model/xenocanto_dataset
python download_xenocanto.py --metadata-only --max-pages 1 --limit-per-species 5
```

This writes:

```text
metadata/pages/*.json
metadata/recordings.jsonl
metadata/summary.json
```

## Download Audio

Download every available recording returned by the API for the target species, filtered by upload date on or before 2025-04-01:

```bash
cd biodcase_model/xenocanto_dataset
python download_xenocanto.py --require-uploaded-date
```

For a smaller first pass:

```bash
python download_xenocanto.py --quality A --quality B --max-pages 2 --limit-per-species 50
```

Restrict to one species:

```bash
python download_xenocanto.py --species "Great Tit" --max-pages 1
```

Output layout:

```text
xenocanto_dataset/
├── raw_audio/
│   ├── common_chaffinch/
│   ├── common_chiffchaff/
│   └── ...
└── metadata/
    ├── pages/
    ├── recordings.jsonl
    └── summary.json
```

Downloaded audio and metadata are ignored by Git.

## Preprocess to BioDCASE-style snippets

`preprocess_xenocanto.py` converts raw Xeno-canto recordings into one 3-second snippet per recording using energy-based peak detection adapted from the `rf4423` pipeline:

1. decode input audio;
2. convert to mono;
3. resample to `24,000 Hz`;
4. apply band-pass filtering and adaptive peak detection;
5. extract one `3.0s` snippet centered on the strongest detected peak;
6. fallback to the maximum-energy `3.0s` window if no peak is found;
7. export `.wav` as `16-bit PCM`.

Example:

```bash
cd biodcase_model/xenocanto_dataset
python preprocess_xenocanto.py \
  --input-root /mnt/sda4/datasets/xenocanto_biodcase_2025cutoff_10k \
  --raw-audio-dir /mnt/sda4/datasets/xenocanto_biodcase_2025cutoff_10k/raw_audio \
  --manifest-path /mnt/sda4/datasets/xenocanto_biodcase_2025cutoff_10k/metadata/recordings.jsonl \
  --output-dir /mnt/sda4/datasets/xenocanto_biodcase_2025cutoff_10k_processed_24k_3s

# If disk space is tight, delete each raw file after its snippet is written:
python preprocess_xenocanto.py \
  --input-root /mnt/sda4/datasets/xenocanto_biodcase_2025cutoff_10k \
  --raw-audio-dir /mnt/sda4/datasets/xenocanto_biodcase_2025cutoff_10k/raw_audio \
  --manifest-path /mnt/sda4/datasets/xenocanto_biodcase_2025cutoff_10k/metadata/recordings.jsonl \
  --output-dir /mnt/sda4/datasets/xenocanto_biodcase_2025cutoff_10k_processed_24k_3s \
  --delete-source-after-success
```

Output layout:

```text
..._processed_24k_3s/
├── Common Chaffinch/
├── Common Chiffchaff/
├── ...
└── metadata/
    ├── processed_snippets.jsonl
    └── summary.json
```

For now the script keeps **one snippet per recording**. That keeps the first pretraining dataset simple and avoids exploding the dataset size before we validate the extraction quality.

Do not tune hyperparameters on the hidden evaluation set.
