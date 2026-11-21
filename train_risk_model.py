import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def generate_dataset(n_rows: int = 12000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    scenarios = rng.choice(
        [
            "early_soft_delay",
            "early_unreachable",
            "moderate_arrears",
            "broken_ptp_serial",
            "high_amount_burden",
            "deep_npa_default",
            "disputed_account",
            "edge_boundary_cases",
        ],
        size=n_rows,
        p=[0.22, 0.14, 0.18, 0.14, 0.10, 0.12, 0.05, 0.05],
    )

    dpd = np.zeros(n_rows, dtype=int)
    missed_emi = np.zeros(n_rows, dtype=int)
    emi_amount = np.zeros(n_rows, dtype=float)
    outstanding_amount = np.zeros(n_rows, dtype=float)
    loan_amount = np.zeros(n_rows, dtype=float)
    broken_ptp = np.zeros(n_rows, dtype=int)
    failed_contacts = np.zeros(n_rows, dtype=int)
    has_dispute = np.zeros(n_rows, dtype=int)

    for i, sc in enumerate(scenarios):
        if sc == "early_soft_delay":
            d = int(rng.integers(1, 20))
            m = 1
            emi = float(rng.integers(3000, 25000))
            out = emi * float(rng.uniform(1.0, 1.4))
            loan = max(out * 1.3, emi * int(rng.integers(12, 36)))
            ptp = int(rng.choice([0, 0, 0, 1]))
            fc = int(rng.integers(0, 3))
            disp = 0
        elif sc == "early_unreachable":
            d = int(rng.integers(10, 31))
            m = int(rng.choice([1, 2]))
            emi = float(rng.integers(4000, 30000))
            out = emi * float(rng.uniform(1.2, 2.3))
            loan = max(out * 1.3, emi * int(rng.integers(12, 48)))
            ptp = int(rng.choice([0, 1]))
            fc = int(rng.integers(3, 9))
            disp = 0
        elif sc == "moderate_arrears":
            d = int(rng.integers(31, 61))
            m = int(rng.choice([2, 3]))
            emi = float(rng.integers(6000, 35000))
            out = emi * float(rng.uniform(2.0, 3.6))
            loan = max(out * 1.25, emi * int(rng.integers(18, 48)))
            ptp = int(rng.choice([0, 1, 1]))
            fc = int(rng.integers(1, 6))
            disp = 0
        elif sc == "broken_ptp_serial":
            d = int(rng.integers(35, 89))
            m = int(rng.choice([2, 3, 4]))
            emi = float(rng.integers(8000, 40000))
            out = emi * float(rng.uniform(2.5, 4.8))
            loan = max(out * 1.2, emi * int(rng.integers(18, 60)))
            ptp = int(rng.choice([2, 3, 4]))
            fc = int(rng.integers(2, 8))
            disp = 0
        elif sc == "high_amount_burden":
            d = int(rng.integers(25, 75))
            m = int(rng.choice([2, 3, 4, 5]))
            emi = float(rng.integers(25000, 85000))
            out = emi * float(rng.uniform(3.0, 6.2))
            loan = max(out * 1.15, emi * int(rng.integers(24, 60)))
            ptp = int(rng.choice([0, 1, 2]))
            fc = int(rng.integers(1, 6))
            disp = 0
        elif sc == "deep_npa_default":
            d = int(rng.integers(90, 240))
            m = int(rng.integers(4, 10))
            emi = float(rng.integers(10000, 60000))
            out = emi * float(rng.uniform(4.2, 9.5))
            loan = max(out * 1.05, emi * int(rng.integers(24, 60)))
            ptp = int(rng.integers(1, 5))
            fc = int(rng.integers(4, 14))
            disp = 0
        elif sc == "disputed_account":
            d = int(rng.integers(5, 95))
            m = int(rng.integers(1, 5))
            emi = float(rng.integers(5000, 45000))
            out = emi * float(rng.uniform(1.0, 4.5))
            loan = max(out * 1.2, emi * int(rng.integers(12, 48)))
            ptp = int(rng.choice([0, 1]))
            fc = int(rng.integers(0, 5))
            disp = 1
        else:  # edge_boundary_cases
            d = int(rng.choice([0, 1, 30, 31, 60, 61, 89, 90, 91, 180]))
            m = int(rng.choice([0, 1, 2, 3, 4, 8]))
            emi = float(rng.choice([1500, 5000, 20000, 50000, 100000]))
            out = emi * float(rng.choice([0.5, 1.0, 1.5, 3.0, 5.0, 8.0]))
            loan = max(out * 1.0, emi * int(rng.integers(6, 60)))
            ptp = int(rng.choice([0, 1, 2, 4]))
            fc = int(rng.choice([0, 1, 5, 12]))
            disp = int(rng.choice([0, 1]))

        dpd[i] = d
        missed_emi[i] = m
        emi_amount[i] = round(emi, 2)
        outstanding_amount[i] = round(out, 2)
        loan_amount[i] = round(loan, 2)
        broken_ptp[i] = ptp
        failed_contacts[i] = fc
        has_dispute[i] = disp

    # Amount & Skipped EMIs interaction features
    skipped_emi_amount = np.round(missed_emi * emi_amount, 2)
    outstanding_to_emi_ratio = np.round(outstanding_amount / np.maximum(emi_amount, 1.0), 3)
    skipped_to_outstanding_ratio = np.round(
        np.minimum(1.5, skipped_emi_amount / np.maximum(outstanding_amount, 1.0)), 3
    )
    principal_exposure_ratio = np.round(
        np.minimum(1.0, outstanding_amount / np.maximum(loan_amount, 1.0)), 3
    )

    # Calibrated target risk score (0-100)
    dpd_pts = np.where(dpd >= 90, 25,
              np.where(dpd >= 61, 20,
              np.where(dpd >= 31, 14,
              np.where(dpd >= 1, 6, 0))))
    emi_pts = np.minimum(15, np.round((missed_emi / 3.0) * 15))
    burden_pts = np.minimum(
        20,
        np.round(
            skipped_to_outstanding_ratio * 14.0
            + np.where(skipped_emi_amount >= 100000, 6,
              np.where(skipped_emi_amount >= 50000, 4,
              np.where(skipped_emi_amount >= 20000, 2, 0)))
        ),
    )
    ratio_pts = np.where(outstanding_to_emi_ratio >= 5.0, 15,
                np.where(outstanding_to_emi_ratio >= 3.0, 11,
                np.where(outstanding_to_emi_ratio >= 1.5, 8, 3)))
    ptp_pts = np.minimum(15, broken_ptp * 8)
    contact_pts = np.minimum(10, failed_contacts * 2)
    interaction_boost = np.where((skipped_emi_amount >= 50000) & (broken_ptp >= 2), 4.0, 0.0)
    noise = rng.normal(0, 1.5, size=n_rows)

    risk_score = np.clip(
        np.round(dpd_pts + emi_pts + burden_pts + ratio_pts + ptp_pts + contact_pts + interaction_boost + noise, 1),
        0, 100,
    )

    route_track = np.where(has_dispute == 1, "DISPUTED",
                  np.where(risk_score >= 60.0, "FIELD_TRACK", "DIGITAL_TRACK"))

    return pd.DataFrame({
        "account_number": [f"ACC-{100000 + i}" for i in range(n_rows)],
        "scenario": scenarios,
        "loan_amount": loan_amount,
        "emi_amount": emi_amount,
        "outstanding_amount": outstanding_amount,
        "dpd": dpd,
        "missed_emi_count": missed_emi,
        "skipped_emi_amount": skipped_emi_amount,
        "outstanding_to_emi_ratio": outstanding_to_emi_ratio,
        "skipped_to_outstanding_ratio": skipped_to_outstanding_ratio,
        "principal_exposure_ratio": principal_exposure_ratio,
        "broken_promise_count": broken_ptp,
        "failed_contact_attempts": failed_contacts,
        "has_active_dispute": has_dispute,
        "risk_score": risk_score,
        "route_track": route_track,
    })


FEATURE_COLS = [
    "loan_amount",
    "emi_amount",
    "outstanding_amount",
    "dpd",
    "missed_emi_count",
    "skipped_emi_amount",
    "outstanding_to_emi_ratio",
    "skipped_to_outstanding_ratio",
    "principal_exposure_ratio",
    "broken_promise_count",
    "failed_contact_attempts",
    "has_active_dispute",
]


def build_reason_chips(row: pd.Series) -> list[str]:
    chips = []
    if row["dpd"] >= 90:
        chips.append(f"90+ DPD ({int(row['dpd'])}d)")
    elif row["dpd"] >= 61:
        chips.append(f"61-90 DPD ({int(row['dpd'])}d)")
    elif row["dpd"] >= 31:
        chips.append(f"31-60 DPD ({int(row['dpd'])}d)")
    elif row["dpd"] >= 1:
        chips.append(f"1-30 DPD ({int(row['dpd'])}d)")

    if row["missed_emi_count"] >= 3:
        chips.append(f"Default risk: {int(row['missed_emi_count'])} missed EMIs")
    elif row["missed_emi_count"] > 0:
        chips.append(f"{int(row['missed_emi_count'])} missed EMI(s)")

    if row["skipped_emi_amount"] >= 50000:
        chips.append(f"High skipped amount: ₹{int(row['skipped_emi_amount']):,}")
    elif row["skipped_emi_amount"] > 0:
        chips.append(f"Skipped EMIs: ₹{int(row['skipped_emi_amount']):,}")

    if row["broken_promise_count"] > 0:
        chips.append(f"{int(row['broken_promise_count'])} broken promise(s)")

    if row["failed_contact_attempts"] >= 3:
        chips.append(f"{int(row['failed_contact_attempts'])} failed contacts")

    if row["has_active_dispute"] == 1:
        chips.append("Active dispute")

    return chips


if __name__ == "__main__":
    df = generate_dataset(n_rows=12000)
    df.to_csv("recover_iq_ml_dataset_12k.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURE_COLS], df["risk_score"], test_size=0.2, random_state=42
    )

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.06,
        max_depth=6,
        min_samples_leaf=25,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = np.clip(model.predict(X_test), 0, 100)
    print(f"MAE: {mean_absolute_error(y_test, preds):.3f} | R2: {r2_score(y_test, preds):.4f}")

    joblib.dump({"model": model, "features": FEATURE_COLS}, "risk_model.joblib")
    print("Saved risk_model.joblib")
