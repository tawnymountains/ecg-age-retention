"""Calibration by recorded-age band, WCE, older-patient error, Uno's C and IPCW Brier score.

Simulated data only. Two simulated frozen representations of the same patients
are compared: one keeps the full age signal, the other caps it at 65 years, so
a model fitted on it cannot tell older patients apart. For each
representation a Cox readout is fitted on one half of the patients and applied
unchanged to the other half. The script then prints, at a 5-year horizon:

* the signed calibration error Q (eq. 6) in the full cohort, in patients aged
  >= 75 years and in each recorded-age band,
* the weighted calibration error WCE (eq. 7),
* Uno's C and the IPCW Brier score.

Run from the repository root after ``pip install .`` (or with ``PYTHONPATH=src``)::

    python examples/01_survival_evaluation.py
    python examples/01_survival_evaluation.py --plot q_by_age_band.png   # needs matplotlib
"""

from __future__ import annotations

import argparse

import numpy as np

from ecg_age_retention.evaluation import (
    calibration_summary,
    fit_cox_readout,
    ipcw_brier_score,
    select_cox_penalty,
    uno_c,
)
from synthetic_cohort import HORIZON, band_label, simulate_cohort, simulate_representation, split_patients

REPRESENTATIONS = (
    ("full age signal", None),
    ("age signal capped at 65 y", 65.0),
)


def evaluate(x, cohort, fit, evaluation, seed):
    """Fit the Cox readout on ``fit`` patients and evaluate it on ``evaluation`` patients."""
    event_fit, time_fit = cohort.event[fit], cohort.time[fit]
    event_eval, time_eval = cohort.event[evaluation], cohort.time[evaluation]
    selection = select_cox_penalty(x[fit], event_fit, time_fit, HORIZON, random_state=seed)
    readout = fit_cox_readout(x[fit], event_fit, time_fit, alpha=selection.alpha)
    prediction = readout.predict(x[evaluation], HORIZON)
    summary = calibration_summary(event_eval, time_eval, prediction.risk, cohort.age[evaluation], HORIZON)
    c_index = uno_c(event_fit, time_fit, event_eval, time_eval, prediction.linear_predictor, HORIZON)
    brier = ipcw_brier_score(event_fit, time_fit, event_eval, time_eval, prediction.risk, HORIZON)
    return dict(alpha=selection.alpha, summary=summary, uno_c=c_index, brier=brier)


def plot_bands(path, labels, outputs):
    """Save a line plot of Q by recorded-age band (optional, needs matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bands = outputs[0]["summary"].bands.bands
    positions = np.arange(len(bands))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for label, output in zip(labels, outputs):
        ax.plot(positions, output["summary"].bands.q, marker="o", label=label)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels([band_label(lower, upper) for lower, upper in bands])
    ax.set_xlabel("Recorded age (years)")
    ax.set_ylabel("Q (per 1,000 patients)")
    ax.set_title("Simulated data: signed calibration error by age band")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-patients", type=int, default=20000, help="number of simulated patients")
    parser.add_argument("--seed", type=int, default=7, help="seed of the simulation and of the penalty folds")
    parser.add_argument("--plot", metavar="FILE", help="save a plot of Q by age band to FILE (needs matplotlib)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    cohort = simulate_cohort(args.n_patients, rng)
    fit, evaluation = split_patients(args.n_patients, (0.5, 0.5), rng)
    labels = [label for label, _ in REPRESENTATIONS]
    outputs = []
    for _, ceiling in REPRESENTATIONS:
        x = simulate_representation(cohort, rng, age_ceiling=ceiling)
        outputs.append(evaluate(x, cohort, fit, evaluation, args.seed))

    print("Synthetic demonstration: survival evaluation of two simulated representations")
    print("Simulated data; every value below is illustrative and does not come from the paper.")
    print()
    print(
        f"Patients: {fit.size} for fitting the Cox readout, {evaluation.size} for evaluation; "
        f"horizon H = {HORIZON:g} years"
    )
    print("Q = Kaplan-Meier observed risk minus mean predicted risk, per 1,000 patients")
    print("(positive = risk underestimated, negative = risk overestimated).")
    print()
    header = f"{'Representation':<28}{'penalty':>9}{'Q_full':>9}{'Q_>=75':>9}{'WCE':>8}{'Uno C':>9}{'Brier':>9}"
    print(header)
    print("-" * len(header))
    for label, output in zip(labels, outputs):
        summary = output["summary"]
        print(
            f"{label:<28}{output['alpha']:>9g}{summary.q_full:>+9.1f}{summary.q_older:>+9.1f}"
            f"{summary.wce:>8.1f}{output['uno_c']:>9.4f}{output['brier']:>9.4f}"
        )
    print()

    bands = outputs[0]["summary"].bands
    print("Q by recorded-age band (evaluation patients)")
    header = f"{'Age band':<10}{'patients':>9}{'deaths<=H':>11}{'observed':>10}" + "".join(
        f"{'Q, ' + label:>30}" for label in labels
    )
    print(header)
    print("-" * len(header))
    for k, (lower, upper) in enumerate(bands.bands):
        row = (
            f"{band_label(lower, upper):<10}{bands.n[k]:>9d}{bands.n_events_by_horizon[k]:>11d}"
            f"{1000 * bands.observed_risk[k]:>10.1f}"
        )
        row += "".join(f"{output['summary'].bands.q[k]:>+30.1f}" for output in outputs)
        print(row)
    print("(observed = Kaplan-Meier observed risk by H, per 1,000 patients)")

    if args.plot:
        plot_bands(args.plot, labels, outputs)
        print()
        print(f"Plot saved to {args.plot}")


if __name__ == "__main__":
    main()
