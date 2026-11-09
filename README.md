# Adaptive Content–Age Gated Fusion for Multimodal Agricultural Stock Movement Prediction

This repository accompanies the manuscript by Fucheng Liu and Chenxue Yang. Adaptive Temporal-Context Gated Fusion (ATGF) uses separate policy and corporate-event content–age encoders to condition a scalar gate between two market-conditioned content paths.

## Models

The PyTorch implementations used for the six-model comparison are in [src/atgf](src/atgf). Earlier feature-construction, training, and analysis scripts are in the parent [src](src) directory.

| Model | Implementation |
| --- | --- |
| Content-Gated Baseline | `SchemeDModel(9, 23, ablation_mode="ordinary")` |
| ATGF | `SchemeDModel(9, 23, ablation_mode="full")` |
| ATGF-NoAge | `AgeControl(variant="noage", seed=1)` retains both context encoders and event presence and zeros both explicit age inputs |
| Direct Content–Age Gate | `AgeControl(variant="direct", seed=1)` supplies content representations, encoded ages, and event presence directly to the gate |
| Compact PatchTST | `CompactPatchTSTClassifier()` |
| Compact iTransformer | `CompactITransformerClassifier()` |

From the repository root, with PyTorch installed, the following example runs ATGF on synthetic inputs:

```python
import torch
from src.atgf.scheme_d_model import SchemeDModel

model = SchemeDModel(9, 23, ablation_mode="full").eval()
with torch.no_grad():
    result = model(
        torch.zeros(2, 1, 6, 60), torch.zeros(2, 9), torch.zeros(2, 23),
        days_since_policy=torch.tensor([10., 20.]),
        days_since_event=torch.tensor([0., 3.]),
        has_available_event=torch.tensor([0., 1.]),
    )
print(result["logits"].shape)  # torch.Size([2, 2])
```

## Evaluation in the manuscript

The development and external panels each contain four stocks. Per model and panel, the study averages 240 runs across four stocks, three horizons (5, 10, and 20 trading days), four policy-cycle folds (2022–2025), and five seeds (1, 7, 21, 42, 2024). Rolling splits enforce target-date boundaries between training, validation, and testing. Both panels had already been examined when NoAge and Direct were designed.

Positive-class F1 and Matthews correlation coefficient (MCC) are reported for all six models. Average precision (AP) and predicted-positive rate (PPR) are also reported for the four gating configurations.

## Data availability

**ATGF Code for Agricultural Stock Prediction** is the repository title used in the submission's research-data entry. The [numeric market-feature sample](data/jifeng_numeric_sample.csv) contains the first 20 trading observations for Jifeng Technology in 2022. Its columns are trading date (YYYYMMDD), one-day return, five-day volatility, and closing price divided by its five-day moving average. The sample illustrates the feature format. Reproducing the panel results requires the full aligned experimental inputs and evaluation workflow.

For research access to the text, images, or associated annotations, please email [e1303008@u.nus.edu](mailto:e1303008@u.nus.edu). Access is subject to the source providers' redistribution terms.

## Citation

Fucheng Liu and Chenxue Yang. *Adaptive Content–Age Gated Fusion for Multimodal Agricultural Stock Movement Prediction*. Manuscript.

## License

See [LICENSE](LICENSE) for the code license.
