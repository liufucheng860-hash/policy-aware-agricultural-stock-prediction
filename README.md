# Interpretable Policy-Aware Multimodal Learning for Multi-Horizon Agricultural Stock Movement Prediction

This repository provides the source code for the paper:

> **Interpretable Policy-Aware Multimodal Learning for Multi-Horizon Agricultural Stock Movement Prediction**

The project studies agricultural-related stock movement prediction using a policy-aware multimodal learning framework. The framework integrates price-pattern image representations, macro-level agricultural policy features, and micro-level corporate event features, and evaluates their effectiveness across multiple prediction horizons.

## Overview

Stock movement prediction is challenging because financial markets are noisy, nonlinear, and affected by heterogeneous information sources. This project focuses on agricultural-related Chinese listed companies, whose stock movements may be influenced by both market patterns and agricultural policy signals.

The proposed framework includes:

* Image-based stock price representation;
* Macro-level agricultural policy feature construction;
* Micro-level corporate event feature construction;
* Multimodal prediction under 5-day, 10-day, and 20-day horizons;
* Static concatenation and softmax-gated dynamic fusion;
* Evaluation using Accuracy, Precision, Recall, F1-score, and Matthews Correlation Coefficient (MCC);
* Policy-event-aligned modality activation analysis.

## Repository Content

This repository currently contains the **source code only**.

Raw data, processed datasets, model outputs, final tables, and figures are not included in this repository.

The source code covers the following components:

```text
Feature construction
Model training
Baseline training
Evaluation metrics
MCC calculation
Dynamic modality-weight analysis
Policy-event-aligned analysis
```

## Code Structure

The repository contains scripts for:

```text
build_macro_features.py          Macro-policy feature construction
build_micro_features.py          Micro-event feature construction
build_price_images.py            Price-pattern image construction
train_image_only.py             Image-only model training
train_macro_only.py             Macro-only model training
train_micro_only.py             Micro-only model training
train_image_macro.py             Image+Macro model training
train_image_micro.py             Image+Micro model training
train_macro_micro.py             Macro+Micro model training
train_image_macro_micro.py       Full multimodal model training
train_multiple_baseline.py      Sequential baseline models
analyze_dynamic_weight.py        Dynamic fusion weight analysis
analyze_policy_before_after.py         Policy-event-aligned modality activation analysis
```

The exact file names may vary slightly depending on the local implementation version.

## Installation

Clone the repository:

```bash
git clone https://github.com/liufucheng860-hash/policy-aware-agricultural-stock-prediction.git
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

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Availability

The datasets used in this study were collected and organized by the authors. They include historical market data, agricultural policy-related information, and firm-level corporate event features for selected agricultural-related listed companies.

The experimental datasets are not publicly released in this repository due to source-specific access and redistribution considerations. They may be made available from the corresponding author upon reasonable request for academic research purposes.

Contact: e1303008@u.nus.edu

## Experimental Setting

The main experiments are designed around four agricultural-related listed companies:

```text
Jifeng Technology
First Tractor Co., Ltd.
Xingguang Agricultural Machinery
Longping High-Tech
```

The prediction horizons are:

```text
5-day
10-day
20-day
```

The evaluated modality configurations include:

```text
Image-only
Macro-only
Micro-only
Image+Macro
Image+Micro
Macro+Micro
Image+Macro+Micro
```

Sequential baseline models include:

```text
LSTM
GRU
Transformer Encoder
```

## Evaluation Metrics

The experiments use the following classification metrics:

```text
Accuracy
Precision
Recall
F1-score
Matthews Correlation Coefficient (MCC)
```

MCC is included to provide a more reliable evaluation under class imbalance and potential prediction-collapse behavior.

## Reproducibility Notes

Because raw data and processed datasets are not included, the repository is not intended to provide a one-command full reproduction package at the current stage.

Instead, it provides the source-code implementation of the main experimental workflow, including feature construction, model training, evaluation, and policy-event analysis.

A full reproduction requires users to prepare the required input data according to the data construction procedures described in the paper.

## Citation

If you use this code, please cite the associated paper after publication.

```bibtex
@article{liu2026policyaware,
  title={Interpretable Policy-Aware Multimodal Learning for Multi-Horizon Agricultural Stock Movement Prediction},
  author={Liu, Fucheng and Yang, Chenxue},
  year={2026},
  note={Manuscript in preparation}
}
```

## License

This repository is released for academic and research purposes. Please refer to the `LICENSE` file for details.

## Disclaimer

This repository is intended for academic research only. The code and results do not constitute investment advice, trading advice, or financial recommendations.
