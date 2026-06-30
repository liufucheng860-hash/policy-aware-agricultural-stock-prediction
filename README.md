# Interpretable Policy-Aware Multimodal Learning for Multi-Horizon Agricultural Stock Movement Prediction

This repository contains the code, experimental configurations, selected processed outputs, and reproducibility materials for the paper:

> **Interpretable Policy-Aware Multimodal Learning for Multi-Horizon Agricultural Stock Movement Prediction**

## Overview

This study proposes an interpretable policy-aware multimodal learning framework for agricultural-related stock movement prediction.

The framework integrates three heterogeneous information sources:

* **Image-based price-pattern representations**, constructed from historical trading information and technical indicators;
* **Macro-level agricultural policy features**, constructed from major agricultural policy documents;
* **Micro-level corporate event features**, constructed from firm-specific corporate announcements.

The model evaluates multimodal prediction performance across three forecasting horizons:

* **5-day horizon:** short-term prediction;
* **10-day horizon:** transitional prediction;
* **20-day horizon:** medium-term prediction.

In addition to prediction performance, the framework uses adaptive modality fusion and policy-event-aligned modality activation analysis to examine how the contribution of price, policy, and event information changes across stocks, horizons, and policy conditions.

## Main Contributions

This repository supports the following research components:

1. Policy-aware multimodal stock movement prediction for agricultural-related Chinese listed companies;
2. Multi-horizon analysis of price images, macro-policy information, and micro-event information;
3. Comparison of Image-only, Macro-only, Micro-only, dual-modality, and Full Model configurations;
4. Comparison with sequential deep-learning baselines, including LSTM, GRU, and Transformer Encoder models;
5. Evaluation using Accuracy, Precision, Recall, F1-score, and Matthews Correlation Coefficient (MCC);
6. Adaptive gated fusion for modality-level interpretability;
7. Policy-event-aligned modality activation analysis around major agricultural policy releases.

## Repository Structure

```text
policy-aware-agricultural-stock-prediction/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── data_preprocessing.py
│   ├── build_price_images.py
│   ├── build_macro_features.py
│   ├── build_micro_features.py
│   ├── train_multimodal.py
│   ├── train_baselines.py
│   ├── evaluate_metrics.py
│   ├── policy_event_analysis.py
│   └── plot_figures.py
│
├── configs/
│   ├── main_experiment.yaml
│   ├── model_config.yaml
│   └── horizons.yaml
│
├── data/
│   ├── README.md
│   ├── sample/
│   └── processed/
│
├── outputs/
│   ├── tables/
│   ├── figures/
│   ├── sample_results/
│   └── final_mcc_tables/
│
├── notebooks/
│   └── reproduction_demo.ipynb
│
└── paper/
    ├── manuscript.pdf
    └── supplementary_material.pdf
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/policy-aware-agricultural-stock-prediction.git
cd policy-aware-agricultural-stock-prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

For Windows:

```bash
.venv\Scripts\activate
```

For macOS or Linux:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Data Description

The study uses three categories of data:

| Data Category            | Description                                                                  | Main Purpose                                              |
| ------------------------ | ---------------------------------------------------------------------------- | --------------------------------------------------------- |
| Market data              | Historical open, high, low, close, volume, returns, and technical indicators | Construction of image-based price-pattern representations |
| Agricultural policy data | Major agricultural policy documents, including annual No.1 Central Documents | Construction of macro-level policy features               |
| Corporate event data     | Publicly disclosed corporate announcements and company-related events        | Construction of micro-level event features                |

The experiments cover four Chinese agricultural-related listed companies:

| Stock                            |      Code | Main Business                                                      |
| -------------------------------- | --------: | ------------------------------------------------------------------ |
| Jifeng Technology                | 300022.SZ | Agricultural machinery distribution and smart agriculture services |
| First Tractor Co., Ltd.          | 601038.SH | Agricultural machinery and tractor manufacturing                   |
| Xingguang Agricultural Machinery | 603789.SH | Agricultural machinery equipment manufacturing                     |
| Longping High-Tech               | 000998.SZ | Seed technology and agricultural biotechnology                     |

## Data Availability

The study uses market data, publicly disclosed corporate announcements, and agricultural policy documents obtained from external sources.

Due to source-specific access conditions, redistribution restrictions, and possible licensing limitations, complete raw datasets are not redistributed in this repository.

This repository provides:

* Data-processing scripts;
* Feature-construction procedures;
* Sample input files;
* Experimental configurations;
* Selected processed outputs;
* Final tables and figures used in the paper.

Researchers who wish to reproduce the workflow should obtain the original market and announcement data from the relevant data providers and follow the preprocessing procedures in this repository.

## Experimental Settings

The main experiments use a chronological data split:

```text
Training set:   70%
Validation set: 15%
Test set:       15%
```

The stock movement label is defined as:

```text
y = 1, if the future return over horizon h is positive
y = 0, otherwise
```

The prediction horizons are:

```text
5-day, 10-day, and 20-day
```

The reported main results use random seed `42`. Additional seed robustness experiments use:

```text
1, 7, 21, 42, and 2024
```

## Model Configurations

The following modality configurations are evaluated:

```text
Image-only
Macro-only
Micro-only
Image+Macro
Image+Micro
Macro+Micro
Image+Macro+Micro (Full Model)
```

The following sequential baseline models are included:

```text
LSTM
GRU
Transformer Encoder
```

## Reproduction Workflow

### 1. Preprocess market and textual data

```bash
python src/data_preprocessing.py
```

### 2. Construct image-based price representations

```bash
python src/build_price_images.py
```

### 3. Construct macro-policy features

```bash
python src/build_macro_features.py
```

### 4. Construct micro-event features

```bash
python src/build_micro_features.py
```

### 5. Run multimodal experiments

```bash
python src/train_multimodal.py --config configs/main_experiment.yaml
```

### 6. Run sequential baseline experiments

```bash
python src/train_baselines.py --config configs/main_experiment.yaml
```

### 7. Calculate evaluation metrics

```bash
python src/evaluate_metrics.py
```

### 8. Run policy-event modality activation analysis

```bash
python src/policy_event_analysis.py
```

### 9. Generate tables and figures

```bash
python src/plot_figures.py
```

## Evaluation Metrics

The experiments report the following classification metrics:

* Accuracy;
* Precision;
* Recall;
* F1-score;
* Matthews Correlation Coefficient (MCC).

MCC is included because it provides an additional reliability measure under potential class imbalance and prediction-collapse behavior.

## Key Findings

The empirical results indicate that:

* Multimodal information can provide incremental predictive value, but its effectiveness is not uniform across all stocks and horizons;
* Image+Micro tends to perform strongly under short-term and transitional prediction settings;
* Image+Macro tends to provide stronger medium-term F1 performance;
* Full multimodal fusion does not consistently outperform simpler two-modality configurations;
* Some settings with non-negligible F1 scores may still show weak MCC values, indicating limited class-balanced predictive reliability;
* Adaptive fusion weights provide interpretable evidence on how price, macro-policy, and micro-event information contribute under different market conditions;
* Policy-sensitive stocks may shift attention from image-based market representations toward textual policy and event information after major agricultural policy releases.

## Output Files

The main output folders contain:

```text
outputs/tables/
```

Final LaTeX-ready experimental tables.

```text
outputs/figures/
```

Figures used in the manuscript.

```text
outputs/final_mcc_tables/
```

Updated experimental tables containing MCC results and consistency-check reports.

```text
outputs/sample_results/
```

Selected sample outputs for demonstration and reproducibility checking.

## Citation

If you use this repository, please cite the associated paper after publication.

```bibtex
@article{liu2026policyaware,
  title={Interpretable Policy-Aware Multimodal Learning for Multi-Horizon Agricultural Stock Movement Prediction},
  author={Liu, Fucheng},
  year={2026},
  note={Manuscript in preparation}
}
```

## License

This repository is released under the MIT License. See the `LICENSE` file for details.

## Contact

For questions regarding the code, experiment setup, or reproducibility materials, please contact:

```text
e1303008@u.nus.edu
```

## Disclaimer

This repository is intended for academic and research purposes only. The methods and results presented here do not constitute investment advice, trading advice, or financial recommendations.
