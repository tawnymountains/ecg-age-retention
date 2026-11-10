"""Global age-R2, RankMe and rank agreement across simulated representations.

Simulated data only. Five simulated frozen representations of the same patients
differ in how much of the age signal they keep (capped at different ages or
not capped), in the noise on a second prognostic factor and in the scale of
their nuisance dimensions. The patients are split into a Cox-fit set, an
age-probe set and an evaluation set. For every representation the script
computes

* global age-R2 (eq. 4) from the pooled out-of-fold predictions of a nested,
  patient-grouped ridge age probe on the age-probe set,
* RankMe on the age-probe set,
* Uno's C, the full-cohort absolute calibration error |Q_full| and the
  older-patient absolute risk error |Q_>=75| (eq. 6) of a Cox readout fitted on
  the Cox-fit set and applied unchanged to the evaluation set,

and then the Spearman rank agreement of each metric with -|Q_>=75| across the
representations (eq. 8).

Run from the repository root after ``pip install .`` (or with ``PYTHONPATH=src``)::

    python examples/02_representation_metrics.py --inner-folds <k>
"""

from __future__ import annotations

import argparse

import numpy as np

from ecg_age_retention.evaluation import calibration_summary, fit_cox_readout, select_cox_penalty, uno_c
from ecg_age_retention.representation import global_age_r2, nested_age_probe, rank_agreement, rankme
from synthetic_cohort import HORIZON, simulate_cohort, simulate_representation, split_patients

#: (label, age ceiling in years or None, noise SD of the ECG factor, nuisance scale)
REPRESENTATIONS = (
    ("age capped at 55 y", 55.0, 0.3, 1.0),
    ("age capped at 65 y", 65.0, 0.6, 2.0),
    ("age capped at 75 y", 75.0, 0.3, 0.5),
    ("age capped at 85 y", 85.0, 1.0, 1.0),
    ("full age signal", None, 0.6, 1.5),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-patients", type=int, default=20000, help="number of simulated patients")
    parser.add_argument("--seed", type=int, default=7, help="seed of the simulation and of the penalty folds")
    parser.add_argument(
        "--inner-folds",
        type=int,
        required=True,
        help="number of inner folds used by the age probe to select its penalty (required, no default)",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    cohort = simulate_cohort(args.n_patients, rng)
    fit, probe, evaluation = split_patients(args.n_patients, (0.4, 0.2, 0.4), rng)

    rows = []
    for label, ceiling, ecg_noise, nuisance_scale in REPRESENTATIONS:
        x = simulate_representation(
            cohort, rng, age_ceiling=ceiling, ecg_noise=ecg_noise, nuisance_scale=nuisance_scale
        )
        # Age-information retention on the age-probe set (one ECG per simulated patient).
        age_probe = nested_age_probe(
            x[probe], cohort.age[probe], cohort.patient_id[probe], inner_folds=args.inner_folds
        )
        age_r2 = global_age_r2(cohort.age[probe], age_probe.oof_prediction)
        effective_rank = rankme(x[probe])
        # Independent survival evaluation: Cox readout fitted on the Cox-fit set only.
        event_fit, time_fit = cohort.event[fit], cohort.time[fit]
        event_eval, time_eval = cohort.event[evaluation], cohort.time[evaluation]
        alpha = select_cox_penalty(x[fit], event_fit, time_fit, HORIZON, random_state=args.seed).alpha
        readout = fit_cox_readout(x[fit], event_fit, time_fit, alpha=alpha)
        prediction = readout.predict(x[evaluation], HORIZON)
        summary = calibration_summary(event_eval, time_eval, prediction.risk, cohort.age[evaluation], HORIZON)
        c_index = uno_c(event_fit, time_fit, event_eval, time_eval, prediction.linear_predictor, HORIZON)
        rows.append(
            dict(
                label=label,
                age_r2=age_r2,
                rankme=effective_rank,
                uno_c=c_index,
                abs_q_full=abs(summary.q_full),
                older_error=abs(summary.q_older),
            )
        )

    print("Synthetic demonstration: representation metrics across simulated representations")
    print("Simulated data; every value below is illustrative and does not come from the paper.")
    print()
    print(
        f"Patients: {fit.size} Cox-fit, {probe.size} age-probe, {evaluation.size} evaluation; "
        f"horizon H = {HORIZON:g} years"
    )
    print()
    header = f"{'Representation':<22}{'age-R2':>9}{'RankMe':>9}{'Uno C':>9}{'|Q_full|':>10}{'|Q_>=75|':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['label']:<22}{row['age_r2']:>9.3f}{row['rankme']:>9.2f}{row['uno_c']:>9.4f}"
            f"{row['abs_q_full']:>10.1f}{row['older_error']:>10.1f}"
        )
    print("(|Q| in absolute risk per 1,000 patients)")
    print()

    older_error = [row["older_error"] for row in rows]
    print(f"Spearman rank agreement with -|Q_>=75| across the {len(rows)} representations (eq. 8):")
    for name, key, higher_is_better in (
        ("global age-R2", "age_r2", True),
        ("Uno's C", "uno_c", True),
        ("RankMe", "rankme", True),
        ("|Q_full| (negated)", "abs_q_full", False),
    ):
        rho = rank_agreement([row[key] for row in rows], older_error, higher_is_better=higher_is_better)
        print(f"  {name:<20}{rho:>+7.3f}")


if __name__ == "__main__":
    main()
