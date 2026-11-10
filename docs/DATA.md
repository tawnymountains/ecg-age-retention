# Data access and input conventions

No patient-level data, extracted features or trained weights are distributed with this repository. This page lists where the datasets used in the paper can be obtained and how the toolkit expects its inputs.

## Dataset access

### CODE-15%

1. Open Zenodo record 4916206: <https://zenodo.org/records/4916206>.
2. Download the exam table (`exams.csv`) and the archives `exams_part0.zip` to `exams_part17.zip`, and extract the tracing files `exams_part*.hdf5`.
3. Each tracing is stored as a 4096 × 12 array (samples × leads) in the lead order I, II, III, aVR, aVL, aVF, V1–V6. Transpose it to 12 × 4096 before passing it to the preprocessing functions or to the encoder.

### MIMIC-IV-ECG and MIMIC-IV

1. MIMIC-IV-ECG: <https://physionet.org/content/mimic-iv-ecg/1.0/>. Follow the license terms shown on the project page and download the files.
2. MIMIC-IV: <https://physionet.org/content/mimiciv/>. Access requires a credentialed PhysioNet account: create an account, complete the credentialing application and the required training, sign the data use agreement of the project, and then download the files.
3. In the paper, mortality outcomes of MIMIC-IV-ECG patients come from the linked MIMIC-IV clinical database.

### External ECG cohorts

| Cohort | Source |
| --- | --- |
| PTB-XL | <https://physionet.org/content/ptb-xl/> |
| Chapman–Shaoxing–Ningbo | <https://physionet.org/content/ecg-arrhythmia/> |
| Georgia, CPSC-2018 and CPSC-Extra | Training data of the PhysioNet/CinC Challenge 2020: <https://physionet.org/content/challenge-2020/> |

Download the records from the project pages. They are stored in WFDB format (a header file per record together with its signal file) and can be read with a WFDB reader such as the `wfdb` Python package, which is not a dependency of this toolkit.

## Input conventions

### Survival outcomes and predictions (NumPy)

| Argument | Shape | Convention |
| --- | --- | --- |
| `event` | `(n,)` | Boolean or 0/1; `True`/1 = death observed, `False`/0 = censored |
| `time` | `(n,)` | Follow-up time, finite and non-negative, in the same unit as `horizon` |
| `horizon` | scalar | Prediction horizon H |
| `predicted_risk` | `(n,)` | Predicted probability of death by H, for example `CoxReadout.predict(...).risk` |
| `risk_score` | `(n,)` | Score for Uno's C, higher = higher risk, for example `CoxReadout.predict(...).linear_predictor` |
| `age` | `(n,)` | Recorded age at ECG acquisition, in years |
| `train_event`, `train_time` | `(m,)` | Outcomes of the patients the Cox model was fitted on; they define the censoring distribution used by Uno's C, the IPCW Brier score and the IPCW log-loss |

- Recorded-age bands are half-open intervals `[lower, upper)`. The default bands 15–29, 30–44, 45–59, 60–74, 75–79, 80–84 and ≥85 years are `[15, 30)`, `[30, 45)`, `[45, 60)`, `[60, 75)`, `[75, 80)`, `[80, 85)` and `[85, inf)`. Every patient must fall into exactly one band, so ages below 15 raise an error unless other `bands` are passed. The older-patient subgroup is `age >= 75`.
- Uno's C and the IPCW Brier score weight patients by the Kaplan–Meier censoring survival function of the reference sample (`train_event`, `train_time`). Uno's C evaluates it at the evaluation event times before the horizon, the IPCW Brier score at every evaluation follow-up time and at the horizon. A time beyond the largest reference time raises an error unless the censoring survival has already reached zero there. The IPCW Brier score also requires `min(time) <= horizon < max(time)`.

### Representations (NumPy)

| Argument | Shape | Convention |
| --- | --- | --- |
| `x` (Cox readout) | `(n_patients, n_features)` | Frozen representation, one row per patient. Fit on the Cox-fit patients and apply the fitted readout unchanged to the evaluation patients |
| `features`, `age`, `groups` (age probe) | `(n_ecgs, n_features)`, `(n_ecgs,)`, `(n_ecgs,)` | One row per ECG; `groups` holds the patient identifier, so that the ECGs of one patient never fall on both sides of a split |
| `features` (RankMe) | `(n_rows, n_features)` | Every row contributes to the spectrum |
| `metric`, `older_error` (rank agreement) | `(n_models,)` | One value per model, in the same order; `older_error` is the older-patient absolute risk error \|Q<sub>m,≥75</sub>(H)\| |

Fold membership of the grouped age probe depends on how the group labels are encoded. Use one encoding of the patient identifiers for all models that are compared.

### ECG signals

| Input | Shape and type | Convention |
| --- | --- | --- |
| Recording passed to `reorder_leads` | `(n_leads, n_samples)` | Leads first, one lead name per row; the twelve standard leads are selected and ordered by name |
| Model input | float32 `(batch, 12, n_samples)` | Lead order I, II, III, aVR, aVL, aVF, V1–V6; each lead z-score standardized within its recording and clipped to [−10, 10] (`standardize_leads`); the paper uses 4096 samples |

Recordings of another length can be brought to 4096 samples with `resample_fft`. Apply `zero_nonfinite` before resampling if a recording contains non-finite samples.

### Training inputs (PyTorch)

| Input | Shape | Convention |
| --- | --- | --- |
| Views, masked or noisy inputs, targets | float32 `(batch, 12, n_samples)` | Formed by the caller; the objectives do not generate them |
| `mask` (masked reconstruction) | `(batch, 1, n_samples)` | 1 at masked time samples and 0 elsewhere, shared by all leads |
| `age`, `age2` | float32 `(batch,)` | Recorded age in years; `age2` gives the ages of the second view when they differ from `age` |
| `time`, `event` (Cox batch loss) | `(batch,)` | Follow-up time and event indicator, the event as float 0/1 |
