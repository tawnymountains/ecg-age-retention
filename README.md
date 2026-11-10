# Toward Reliable ECG-based Survival Models via Evaluating Age-Information Retention

ECG-based survival models are usually evaluated by discrimination and calibration in the full cohort, where overestimation in some patients can offset underestimation in others. The paper finds that ECG-based survival models that appear well calibrated in the full cohort can underestimate mortality risk in patients aged 75 years or older, and relates this older-patient risk error to how much age information the frozen ECG representation retains. It proposes global age-R², computed from the pooled out-of-fold predictions of a linear ridge age probe, as a complementary evaluation metric, compares it with the older-patient absolute risk error of an independent Cox readout, and adds auxiliary age supervision, an age-prediction loss, to the original training objective of each model. This repository provides the evaluation metrics, the Cox readout, the ECG encoder with its self-supervised objectives and the auxiliary age head as a Python toolkit, together with examples on simulated data. It is not a pipeline for reproducing the experiments of the paper.

## Framework

The toolkit provides the components of the four steps of the paper's framework:

1. **Frozen ECG representations.** ECG models are trained and then frozen to extract representations. `ecg_age_retention.models` contains the 1D ResNet encoder and its self-supervised objectives, and `ecg_age_retention.data` the signal preprocessing.
2. **Age-information retention.** Global age-R² is computed from the pooled out-of-fold predictions of a nested, patient-grouped linear ridge age probe (`ecg_age_retention.representation`).
3. **Independent survival evaluation.** An L<sub>2</sub>-regularized Cox model is fitted on each frozen representation in one set of patients and evaluated in another: signed calibration error in the full cohort, in recorded-age bands and in patients aged ≥75 years, the weighted calibration error, Uno's C and the IPCW Brier score (`ecg_age_retention.evaluation`). Global age-R² and other selection metrics are compared by their rank agreement with the older-patient absolute risk error (`rank_agreement`).
4. **Auxiliary age supervision.** An age head with a Huber loss is added to the original training objective of a model and discarded after training (`ecg_age_retention.supervision`); the resulting representations are evaluated again with steps 2 and 3.

## What is in this repository

| Path | Contents |
| --- | --- |
| `src/ecg_age_retention/evaluation` | Cox readout on frozen representations (standardization with an optional PCA cap, L<sub>2</sub>-penalized Cox model with Breslow baseline, predicted risk at the horizon, eq. 5) and survival metrics at the horizon: Kaplan–Meier observed risk, signed calibration error Q (eq. 6), weighted calibration error over recorded-age bands (WCE, eq. 7), older-patient absolute risk error, Uno's C and IPCW Brier score |
| `src/ecg_age_retention/representation` | Global age-R² from a nested, patient-grouped ridge age probe (eq. 4), RankMe, and rank agreement with older-patient absolute risk error (eq. 8) |
| `src/ecg_age_retention/models` | 1D ResNet ECG encoder, self-supervised objectives (contrastive, BYOL, VICReg, masked reconstruction, denoising) and a mini-batch Cox partial-likelihood loss; requires PyTorch |
| `src/ecg_age_retention/supervision` | Auxiliary age head, Huber age loss (eqs. 9–10) and a wrapper that adds it to an objective; requires PyTorch |
| `src/ecg_age_retention/data` | ECG preprocessing to the common 12 × 4096 input: lead order, Fourier resampling, per-lead standardization and clipping |
| `examples/` | Runnable demonstrations on simulated data |
| `tests/` | Unit tests |

The evaluation, representation and data modules need only NumPy, SciPy and scikit-learn.

## What is not included

- Trained weights
- Extracted features
- Patient-level data, which cannot be shared; MIMIC-IV requires credentialed PhysioNet access

## Installation

Python 3.10 or newer. From the root of a copy of this repository:

```bash
pip install .               # evaluation, representation and data modules
pip install ".[models]"     # adds PyTorch for the encoder, objectives and age head
pip install ".[examples]"   # adds matplotlib for the optional plot of the first example
```

Without installation, the modules can be used by adding `src` to `PYTHONPATH`.

The NumPy-based tests pass with Python 3.12 (NumPy 1.26, SciPy 1.13, scikit-learn 1.8), and the full test suite passes with Python 3.11 (NumPy 1.26, SciPy 1.17, scikit-learn 1.7, PyTorch 2.11 on CPU). Run the tests with `python -m pytest`, or with `python -m unittest discover -s tests` after installation; the PyTorch tests are skipped when PyTorch is not installed.

## Quick start

Demonstrations on simulated data, run from the repository root:

```bash
python examples/01_survival_evaluation.py      # Q by age band, WCE, older-patient error, Uno's C, IPCW Brier
python examples/02_representation_metrics.py --inner-folds <k>   # global age-R², RankMe and rank agreement
python examples/03_age_supervision_torch.py --lambda-age <weight>   # one CPU step with the age head (PyTorch)
```

Evaluating frozen representations:

```python
from ecg_age_retention.evaluation import (
    calibration_summary, fit_cox_readout, ipcw_brier_score, select_cox_penalty, uno_c,
)
from ecg_age_retention.representation import global_age_r2, nested_age_probe, rank_agreement

H = 5.0  # prediction horizon, in the unit of the follow-up times

# Cox readout fitted on the Cox-fit patients and applied unchanged to the evaluation patients
alpha = select_cox_penalty(x_fit, event_fit, time_fit, H, random_state=seed).alpha
readout = fit_cox_readout(x_fit, event_fit, time_fit, alpha=alpha)
pred = readout.predict(x_eval, H)

cal = calibration_summary(event_eval, time_eval, pred.risk, age_eval, H)
cal.q_full, cal.q_older, cal.wce   # Q_full(H) and Q_>=75(H) (eq. 6), WCE (eq. 7)
uno = uno_c(event_fit, time_fit, event_eval, time_eval, pred.linear_predictor, H)
brier = ipcw_brier_score(event_fit, time_fit, event_eval, time_eval, pred.risk, H)

# Global age-R2 (eq. 4) from pooled out-of-fold predictions on separate age-probe patients
probe = nested_age_probe(x_probe, age_probe, patient_id_probe, inner_folds=k)
age_r2 = global_age_r2(age_probe, probe.oof_prediction)

# Rank agreement across models (eq. 8): one value per model; older error = |Q_>=75(H)|
rho = rank_agreement(age_r2_per_model, older_error_per_model)
```

Auxiliary age supervision (PyTorch):

```python
from ecg_age_retention.models import ContrastiveObjective, ResNet1dEncoder
from ecg_age_retention.supervision import AgeHead, AgeSupervisedObjective

model = AgeSupervisedObjective(
    ContrastiveObjective(ResNet1dEncoder()), AgeHead(hidden_dim=hidden_dim), lambda_age=lambda_age
)
out = model(view1, view2, age=age)  # views: (batch, 12, 4096); age: (batch,) in years
out.total.backward()                # L_task + lambda_age * L_age (eq. 10)
encoder = model.encoder             # kept after training; the age head is discarded
```

The Cox penalty `alpha`, the fold seed `random_state` of the penalty selection, the number of inner folds `inner_folds` of the age probe, the age-loss weight `lambda_age` and the age-head width `hidden_dim` are required arguments without defaults; `RepresentationTransform` also requires `random_state` when the PCA cap applies. Where the paper does not specify a value, the defaults of the remaining arguments are implementation choices of this toolkit. Array conventions are listed in [docs/DATA.md](docs/DATA.md).

## API overview

| Module | Name | Purpose |
| --- | --- | --- |
| `evaluation` | `kaplan_meier_risk` | Kaplan–Meier observed event risk at the horizon |
| | `calibration_error` | Signed calibration error Q (eq. 6) for all patients or a subgroup |
| | `age_band_calibration`, `weighted_calibration_error` | Q in each recorded-age band and the WCE (eq. 7) |
| | `calibration_summary` | Q in the full cohort and in patients aged ≥75 years, and the WCE |
| | `uno_c` | Uno's C truncated at the horizon |
| | `ipcw_brier_score` | IPCW Brier score at the horizon |
| | `fit_cox_readout`, `CoxReadout` | Representation transform and L<sub>2</sub>-penalized Cox model; `predict` gives the risk at the horizon (eq. 5) |
| | `select_cox_penalty` | Cox penalty selection by cross-validated IPCW log-loss at the horizon |
| | `RepresentationTransform`, `fit_penalized_cox`, `CoxModel`, `breslow_cumulative_hazard`, `ipcw_horizon_log_loss` | Building blocks of the readout |
| `representation` | `nested_age_probe` | Nested, patient-grouped ridge age probe with pooled out-of-fold predictions |
| | `global_age_r2` | Global age-R² (eq. 4) |
| | `ridge_age_predictions` | Ridge age predictions for a grid of penalties |
| | `rankme` | RankMe effective rank of a representation matrix |
| | `rank_agreement` | Spearman rank agreement of a metric with the negated older-patient absolute risk error (eq. 8) |
| `models` | `ResNet1dEncoder` | 1D ResNet encoder: (batch, 12, samples) to (batch, 256) |
| | `ContrastiveObjective`, `BYOLObjective`, `VICRegObjective` | Two-view self-supervised objectives |
| | `MaskedReconstructionObjective`, `DenoisingObjective` | Reconstruction objectives |
| | `CoxObjective`, `cox_ph_loss` | Survival objective with the mini-batch Cox partial likelihood |
| `supervision` | `AgeHead` | Age-prediction head |
| | `age_supervision_loss` | Huber age loss multiplied by λ<sub>age</sub> (eqs. 9–10), averaged over views |
| | `AgeSupervisedObjective` | Adds the auxiliary age loss to an objective from `models` |
| `data` | `reorder_leads`, `resample_fft`, `standardize_leads`, `zero_nonfinite` | Preprocessing to the common 12 × 4096 input |

## Datasets

| Dataset | Link |
| --- | --- |
| CODE-15% | Zenodo record 4916206: <https://zenodo.org/records/4916206> |
| MIMIC-IV-ECG | PhysioNet: <https://physionet.org/content/mimic-iv-ecg/1.0/> |
| MIMIC-IV | PhysioNet, credentialed access: <https://physionet.org/content/mimiciv/> |
| PTB-XL | PhysioNet: <https://physionet.org/content/ptb-xl/> |
| Chapman–Shaoxing–Ningbo | PhysioNet: <https://physionet.org/content/ecg-arrhythmia/> |
| Georgia, CPSC-2018 and CPSC-Extra | PhysioNet/CinC Challenge 2020: <https://physionet.org/content/challenge-2020/> |

Access steps and the input conventions of the toolkit are described in [docs/DATA.md](docs/DATA.md).

## End-to-end baselines

The end-to-end survival models compared in the paper are built on public code: DeepSurv, LogHaz, DeepHit, MTLR and Ince–DeepSurv on [ecg-survival-benchmark](https://github.com/cavalab/ecg-survival-benchmark), and the ECG mortality model of Sun et al. on [ECG_mortality_DL](https://github.com/jeremykid/ECG_mortality_DL). This repository contains no code from these projects.

## Citation

If you use this toolkit, please cite the paper. Citation metadata are provided in [CITATION.cff](CITATION.cff); GitHub also shows them under "Cite this repository".

## License

Released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC 4.0); see [LICENSE](LICENSE). The material may be shared and adapted for non-commercial purposes, with appropriate credit.
