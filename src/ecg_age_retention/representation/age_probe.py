"""Global age-R^2: a nested, patient-grouped ridge probe of age on frozen representations.

A linear ridge model predicts the age recorded at ECG acquisition from a frozen
representation. Every row receives exactly one out-of-fold (OOF) prediction from
a nested, group-aware cross-fitting loop, and the pooled OOF predictions are
scored with the coefficient of determination, eq. (4) of the paper.

Procedure of :func:`nested_age_probe`
-------------------------------------
1. Outer loop: ``sklearn.model_selection.GroupKFold(outer_folds)`` over the group
   labels (one label per patient), so all rows of a patient share one outer fold.
2. Inner loop, inside each outer training set: a fresh
   ``GroupKFold(inner_folds)`` split of the outer-training rows by the same group
   labels. For every inner split, ridge models for all penalties in ``alphas``
   are fitted on the inner training part and scored on the inner validation part
   with ``sklearn.metrics.r2_score``. The penalty with the highest unweighted mean
   inner R^2 is selected; ties go to the first penalty in grid order.
3. A ridge model with the selected penalty is refitted on the whole outer training
   set and predicts the held-out outer fold.
4. After the last outer fold every row has one OOF prediction, and global age-R^2
   is computed once from all of them (:func:`global_age_r2`).

Ridge fits (:func:`ridge_age_predictions`) standardize the features with the
column means and population standard deviations (each plus ``1e-8``) of the rows
they are fitted on, apply the same transform to the query rows, use the training
mean age as an unpenalized intercept, and solve the penalized normal equations
for the whole penalty grid with one symmetric eigendecomposition of the
standardized Gram matrix ``Z^T Z``. The penalty therefore acts on standardized
features.

Fold assignment
---------------
``GroupKFold`` without shuffling is deterministic: it orders the unique group
labels with ``numpy.unique`` and assigns groups, largest first, to the currently
lightest fold. Fold membership therefore depends on how the group labels are
encoded and on the scikit-learn and NumPy versions. Use one fixed encoding of
the group labels for all models that are compared.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

__all__ = [
    "DEFAULT_ALPHA_GRID",
    "NestedAgeProbeResult",
    "ridge_age_predictions",
    "nested_age_probe",
    "global_age_r2",
]

#: Ridge penalties searched by the inner cross-validation: 11 values, 1e-4 to 1e6.
DEFAULT_ALPHA_GRID: NDArray[np.float64] = np.array(
    [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6]
)

#: Offset added to every feature standard deviation before standardization.
_SD_OFFSET = 1e-8


class NestedAgeProbeResult(NamedTuple):
    """Output of :func:`nested_age_probe`.

    Attributes
    ----------
    oof_prediction:
        One out-of-fold age prediction per row, shape ``(n_rows,)``.
    selected_alpha:
        Ridge penalty selected by the inner cross-validation, one per outer fold,
        in outer-fold order.
    outer_fold:
        Index of the outer fold in which each row was held out, shape ``(n_rows,)``.
    """

    oof_prediction: NDArray[np.float64]
    selected_alpha: list[float]
    outer_fold: NDArray[np.int_]


def ridge_age_predictions(
    train_features: ArrayLike,
    train_age: ArrayLike,
    query_features: ArrayLike,
    alphas: ArrayLike = DEFAULT_ALPHA_GRID,
) -> NDArray[np.float64]:
    """Fit standardized ridge age models for a grid of penalties and predict query rows.

    For each penalty ``alpha`` the coefficients solve
    ``(Z^T Z + alpha I) w = Z^T (a - mean(a))``, where ``Z`` holds the training
    features standardized with their own column means and population standard
    deviations (each standard deviation plus ``1e-8``) and ``a`` the training ages.
    The prediction for a query row ``x`` is ``((x - mean) / sd) @ w + mean(a)``.
    All penalties share one eigendecomposition of ``Z^T Z``.

    Parameters
    ----------
    train_features:
        Representation of the fitting rows, shape ``(n_train, n_features)``.
    train_age:
        Recorded age of the fitting rows, shape ``(n_train,)``.
    query_features:
        Representation of the rows to predict, shape ``(n_query, n_features)``.
    alphas:
        Ridge penalties, shape ``(n_alphas,)``.

    Returns
    -------
    numpy.ndarray
        Predicted ages, shape ``(n_query, n_alphas)``; column ``j`` uses ``alphas[j]``.
    """
    train_features = np.asarray(train_features, dtype=np.float64)
    train_age = np.asarray(train_age, dtype=np.float64)
    query_features = np.asarray(query_features, dtype=np.float64)
    alphas = np.asarray(alphas, dtype=np.float64)
    mean = train_features.mean(0)
    sd = train_features.std(0) + _SD_OFFSET
    age_mean = train_age.mean()
    standardized = (train_features - mean) / sd
    eigenvalues, eigenvectors = np.linalg.eigh(standardized.T @ standardized)
    projected = eigenvectors.T @ (standardized.T @ (train_age - age_mean))
    coefficients = eigenvectors @ (
        projected[:, None] / (eigenvalues[:, None] + alphas[None, :])
    )
    return ((query_features - mean) / sd) @ coefficients + age_mean


def nested_age_probe(
    features: ArrayLike,
    age: ArrayLike,
    groups: ArrayLike,
    *,
    inner_folds: int,
    outer_folds: int = 5,
    alphas: ArrayLike = DEFAULT_ALPHA_GRID,
) -> NestedAgeProbeResult:
    """Out-of-fold age predictions from a nested, group-aware ridge age probe.

    Implements the nested cross-fitting described in the module docstring: an
    outer ``GroupKFold(outer_folds)`` produces the held-out folds, and inside each
    outer training set a separate ``GroupKFold(inner_folds)`` split selects the
    ridge penalty by the highest mean inner-fold R^2. Pass the returned
    ``oof_prediction`` to :func:`global_age_r2` to obtain global age-R^2 (eq. 4).

    Parameters
    ----------
    features:
        Frozen representation, shape ``(n_rows, n_features)``; converted to float64.
    age:
        Age at ECG acquisition, shape ``(n_rows,)``; converted to float64.
    groups:
        Group label of every row (the patient identifier), shape ``(n_rows,)``.
        Rows with the same label never fall on both sides of an outer or inner split.
    inner_folds:
        Number of inner ``GroupKFold`` splits used to select the ridge penalty
        within each outer training set. Required; there is no default.
    outer_folds:
        Number of outer ``GroupKFold`` splits (five-fold cross-fitting by default).
    alphas:
        Ridge penalties searched by the inner cross-validation.

    Returns
    -------
    NestedAgeProbeResult
        ``(oof_prediction, selected_alpha, outer_fold)``.

    Raises
    ------
    FloatingPointError
        If any out-of-fold prediction is not finite.
    """
    features = np.asarray(features, dtype=np.float64)
    age = np.asarray(age, dtype=np.float64)
    groups = np.asarray(groups)
    alphas = np.asarray(alphas, dtype=np.float64)
    prediction = np.full(len(age), np.nan)
    selected_alpha: list[float] = []
    outer_fold = np.full(len(age), -1)
    outer_splitter = GroupKFold(n_splits=outer_folds)
    for fold, (train, held_out) in enumerate(outer_splitter.split(features, age, groups)):
        inner_scores = []
        inner_splitter = GroupKFold(n_splits=inner_folds)
        for inner_train, inner_valid in inner_splitter.split(
            features[train], age[train], groups[train]
        ):
            grid_prediction = ridge_age_predictions(
                features[train[inner_train]],
                age[train[inner_train]],
                features[train[inner_valid]],
                alphas,
            )
            inner_scores.append(
                [
                    r2_score(age[train[inner_valid]], grid_prediction[:, j])
                    for j in range(len(alphas))
                ]
            )
        best = int(np.argmax(np.mean(inner_scores, axis=0)))
        grid_prediction = ridge_age_predictions(
            features[train], age[train], features[held_out], alphas
        )
        prediction[held_out] = grid_prediction[:, best]
        outer_fold[held_out] = fold
        selected_alpha.append(float(alphas[best]))
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Non-finite out-of-fold age prediction.")
    return NestedAgeProbeResult(prediction, selected_alpha, outer_fold)


def global_age_r2(age: ArrayLike, oof_prediction: ArrayLike) -> float:
    """Global age-R^2 from pooled out-of-fold age predictions, eq. (4).

    ``1 - sum_i (a_i - a_hat_i)^2 / sum_i (a_i - a_bar)^2``, where ``a_bar`` is the
    mean age over all rows passed in. Computed with ``sklearn.metrics.r2_score``,
    which, when all ages are identical, returns 1.0 for a perfect prediction and
    0.0 otherwise instead of an undefined value.

    Parameters
    ----------
    age:
        Age at ECG acquisition, shape ``(n_rows,)``.
    oof_prediction:
        Out-of-fold age prediction for the same rows, for example
        ``nested_age_probe(...).oof_prediction``.

    Returns
    -------
    float
        Global age-R^2.
    """
    return float(r2_score(age, oof_prediction))
