# BIODCase Challenge

## About

The goal of the Bioacoustics on Tiny Hardware task is to develop an automatic classifier of birdsong that complies with the resource constraints of low-cost and battery-powered autonomous recording units.


## Description

The next generation of autonomous recording units contains programmable chips, thus offering the opportunity the opportunity to perform BioDCASE tasks. On-device processing has multiple advantages, such as high durability, low latency, and privacy preservation. However, such “tiny hardware” is limited in terms of memory and compute, which calls for the development of original methods in audio content analysis.

In this context, task participants will revisit the well-known problem of automatic detection of birdsong while adjusting their systems to meet the specifications of a commercially available microcontroller. The challenge focuses on detecting the vocalizations of 10 different bird species using the ESP32-S3-Korvo-2 development board.
Photograph of the ESP32-S3-Korvo-2, the "tiny hardware" of BioDCASE task 3.


The primary challenge is striking the optimal balance between classification accuracy and resource usage. While conventional deep learning approaches might achieve high accuracy, they may not be feasible on embedded hardware. Participants must explore techniques such efficient architecture design, and optimized feature extraction to create a solution that performs well within the hardware constraints.

A baseline implementation is provided as a starting point, which participants can modify and improve upon. Solutions will be evaluated based on classification performance, model size, inference time, and memory usage.

Note: We encourage participants to buy the ESP32-S3-Korvo-2 development board to test their solutions. The board is available for purchase on various online platforms (e.g, at DigiKey). However, the competition can be completed without the board, as the evaluation will be performed on a hidden test set using the baseline system by the organizers.
Dataset

The dataset for this year's task uses field recordings from 10 bird species and an additional set of urban backround sounds from Germany:

    11 class labels categorized in folders
    2750 recordings of 3 seconds each...
    audio is sampled at 24 kHz, mono, 16-bit PCM wav files

The dataset is organized as follows:

```
Development_Set/
├── Train/
│   ├── species_1/
|       ├── recording_1.wav
|       ├── recording_2.wav
|       ├── ...
│   ├── species_2/
│   ├── ...
├── Validation/
│   ├── species_1/
│   ├── species_2/
```

The dataset is available for download on Zenodo
- BioDCASE-Tiny 2026 Dataset (254 MB)
 Evaluation and Baseline System

We provide a baseline system that includes a complete pipelines for audio processing, feature extraction, model training, and deployment to the ESP32-S3-Korvo-2 development board. The baseline system is designed to be easily adaptable for participants to build upon.

You can find the baseline system in the GitHub repository: BioDCASE-Tiny 2026 Baseline System
External Data Resources

The competition focuses on the provided 10-bird species dataset. External data use may be regulated according to the official competition rules
Rules and Submissions

Please follow the "Rules and Submission" section in the baseline repository. Your solution will be evaluated on a hidden test set and the scores will be presented in the upcoming results section of the biodcase website!
Citation

If you use the BioDCASE-Tiny framework or dataset in your research, please cite the following:

@misc{biodcase_tiny_2026_repo,
  author = {Walter, Christian and Benhamadi, Yasmine and Seidel, Tom and Carmantini, Giovanni and Kahl, Stefan},
  title = {BioDCASE-Tiny 2026: A Framework for Bird Species Recognition on Resource-Constrained Hardware},
  year = {2026},
  type = {Software},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {https://github.com/birdnet-team/BioDCASE-Tiny-2026},
}

Dataset Citation

@dataset{biodcase_tiny_2026_dataset,
  author = {Kahl, Stefan, and Martin, Ralph},
  title = {BioDCASE 2026 Task 3: Bioacoustics for Tiny Hardware Development Set},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.19453065},
  url = {https://doi.org/10.5281/zenodo.19453065}
}

Support

If you have questions, please use the BioDCASE Google Groups community forum or create an issue in the Github baseline repo: BioDCASE-Tiny 2026 Baseline System.