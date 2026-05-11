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

## Recommended Next Processing Step

After downloading, build a separate preprocessing script that:

1. decodes MP3/FLAC/etc.;
2. resamples to 24 kHz mono;
3. segments long recordings into 3-second windows;
4. selects windows with acoustic activity rather than blindly taking the first 3 seconds;
5. keeps recording-level train/validation splits to avoid leakage;
6. writes processed clips in a BioDCASE-compatible folder layout.

Do not tune hyperparameters on the hidden evaluation set.
