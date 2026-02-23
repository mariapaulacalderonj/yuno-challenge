"""
Apollo Rides - Transaction Anomaly Detection Pipeline

Session-based detection that reconstructs ride payment lifecycles
and cross-references business records (rides, disputes) against
payment events (transactions) to find revenue leaks.

Detects 5 anomaly types:
  1. Duplicate authorizations
  2. Capture amount mismatches (two-source reconciliation)
  3. Ghost refunds (validated against disputes/cancellations table)
  4. Currency conversion discrepancies (justified thresholds per currency)
  5. Abandoned authorizations (completed rides never captured)

Each anomaly includes:
  - Type-specific revenue impact (money_lost vs money_at_risk)
  - Confidence score (0-100%)
  - Actionable recommendation
"""

import json
import logging
from datetime import datetime
import pandas as pd

DATA_DIR = "data"
CAPTURE_MISMATCH_THRESHOLD = 0.10  # 10%

# Currency-specific FX tolerance thresholds (justified by volatility)
# MXN: low volatility (~0.3%/day) -> 2% tolerance
# COP: medium volatility (~0.5%/day) -> 3% tolerance
# BRL: high volatility (~0.6%/day) -> 3% tolerance (tighter than raw volatility
#   suggests, because auth-to-capture gap is typically <1 hour for ride-hailing,
#   not a full trading day — so intraday drift should be well below daily volatility)
FX_TOLERANCE = {"MXN": 0.02, "COP": 0.03, "BRL": 0.03}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("anomaly_detector")


# ---------------------------------------------------------------------------
# Data Loading & Validation
# ---------------------------------------------------------------------------

def load_data():
    """Load all CSV data sources."""
    rides = pd.read_csv(f"{DATA_DIR}/rides.csv")
    transactions = pd.read_csv(f"{DATA_DIR}/transactions.csv")
    disputes = pd.read_csv(f"{DATA_DIR}/disputes_cancellations.csv")
    exchange_rates = pd.read_csv(f"{DATA_DIR}/exchange_rates.csv")
    return rides, transactions, disputes, exchange_rates


def validate_data(rides, transactions, disputes):
    """
    Validate data quality. Log issues, quarantine bad records.
    Returns cleaned dataframes and a validation report.
    """
    report = {"issues": [], "quarantined_rows": 0}

    # Check for nulls in critical transaction fields
    for col in ["ride_id", "amount", "currency"]:
        if col in transactions.columns:
            null_count = transactions[col].isna().sum()
            if null_count > 0:
                report["issues"].append(f"transactions: {null_count} null values in '{col}'")
                transactions = transactions.dropna(subset=[col])
                report["quarantined_rows"] += null_count

    # Check for nulls in critical ride fields
    for col in ["ride_id", "estimated_fare", "actual_fare"]:
        if col in rides.columns:
            null_count = rides[col].isna().sum()
            if null_count > 0:
                report["issues"].append(f"rides: {null_count} null values in '{col}'")
                rides = rides.dropna(subset=[col])
                report["quarantined_rows"] += null_count

    # Check for duplicate transaction IDs
    dup_txns = transactions["transaction_id"].duplicated().sum()
    if dup_txns > 0:
        report["issues"].append(f"transactions: {dup_txns} duplicate transaction_ids")
        transactions = transactions.drop_duplicates(subset=["transaction_id"], keep="first")
        report["quarantined_rows"] += dup_txns

    # Check for captures before authorizations (impossible — data error)
    for ride_id, group in transactions.sort_values(["ride_id", "timestamp"]).groupby("ride_id"):
        auths = group[group["event_type"] == "authorization"]
        captures = group[group["event_type"] == "capture"]
        if len(auths) > 0 and len(captures) > 0:
            if captures["timestamp"].min() < auths["timestamp"].min():
                report["issues"].append(
                    f"ride {ride_id}: capture before authorization (data error)"
                )
                report["quarantined_rows"] += 1

    # Check for orphaned captures (no prior auth)
    rides_with_auth = set(transactions[transactions["event_type"] == "authorization"]["ride_id"].unique())
    rides_with_capture = set(transactions[transactions["event_type"] == "capture"]["ride_id"].unique())
    orphaned = rides_with_capture - rides_with_auth
    if orphaned:
        report["issues"].append(f"{len(orphaned)} rides have captures with no prior authorization")

    # Check for invalid currency codes
    valid_currencies = {"MXN", "COP", "BRL"}
    invalid = set(transactions["currency"].unique()) - valid_currencies
    if invalid:
        report["issues"].append(f"Invalid currency codes: {invalid}")
        transactions = transactions[transactions["currency"].isin(valid_currencies)]

    return rides, transactions, disputes, report


# ---------------------------------------------------------------------------
# Session Reconstruction
# ---------------------------------------------------------------------------

def reconstruct_sessions(rides, transactions, disputes):
    """
    Reconstruct payment lifecycle for each ride as a state machine.
    Groups events by ride_id, orders by timestamp.
    Returns dict: ride_id -> {ride, events[], has_dispute, lifecycle}
    """
    dispute_ride_ids = set(disputes["ride_id"].unique()) if len(disputes) > 0 else set()

    sessions = {}
    for _, ride in rides.iterrows():
        rid = ride["ride_id"]
        ride_txns = transactions[transactions["ride_id"] == rid].sort_values("timestamp")
        event_types = ride_txns["event_type"].tolist()

        # Classify lifecycle pattern
        auth_count = event_types.count("authorization")
        has_capture = "capture" in event_types
        has_void = "void" in event_types
        has_refund = "refund" in event_types

        if auth_count > 1:
            lifecycle = "duplicate_auth"
        elif auth_count == 1 and has_capture and not has_refund:
            lifecycle = "auth_capture"
        elif auth_count == 1 and has_void:
            lifecycle = "auth_void"
        elif has_capture and has_refund:
            lifecycle = "auth_capture_refund"
        elif auth_count == 1 and not has_capture and not has_void:
            lifecycle = "auth_only"
        else:
            lifecycle = "other"

        sessions[rid] = {
            "ride": ride.to_dict(),
            "events": ride_txns.to_dict("records"),
            "has_dispute": rid in dispute_ride_ids,
            "lifecycle": lifecycle,
        }

    return sessions


# ---------------------------------------------------------------------------
# Detector 1: Duplicate Authorizations
# ---------------------------------------------------------------------------

def detect_duplicate_authorizations(sessions):
    """
    Detect rides with multiple approved authorizations.

    Revenue impact (type-specific):
    - Extra auth ALSO captured -> money_lost (double charge)
    - Extra auth NOT captured -> money_at_risk (funds hold)
    """
    anomalies = []

    for ride_id, session in sessions.items():
        events = session["events"]
        auths = [e for e in events if e["event_type"] == "authorization" and e["status"] == "approved"]

        if len(auths) <= 1:
            continue

        ride = session["ride"]
        first_auth = auths[0]
        extra_auths = auths[1:]
        captures = [e for e in events if e["event_type"] == "capture" and e["status"] == "approved"]
        total_captured = sum(c["amount"] for c in captures)
        extra_auth_total = sum(a["amount"] for a in extra_auths)

        # Type-specific revenue impact
        if total_captured > first_auth["amount"] * 1.1:
            impact_category = "money_lost"
            revenue_impact = total_captured - first_auth["amount"]
        else:
            impact_category = "money_at_risk"
            revenue_impact = extra_auth_total

        # Confidence scoring
        time_gap = abs(
            (datetime.fromisoformat(extra_auths[0]["timestamp"])
             - datetime.fromisoformat(first_auth["timestamp"])).total_seconds()
        )
        amounts_same = all(
            abs(a["amount"] - first_auth["amount"]) / max(first_auth["amount"], 0.01) < 0.01
            for a in extra_auths
        )

        if len(auths) >= 3:
            confidence = 98
        elif amounts_same and time_gap < 60:
            confidence = 95
        elif not amounts_same and time_gap < 60:
            confidence = 85
        elif amounts_same:
            confidence = 80
        else:
            confidence = 70

        recommendation = (
            f"Void duplicate authorization(s) for ride {ride_id}. "
            f"{len(extra_auths)} extra auth(s) detected "
            f"({time_gap:.0f}s after first). "
            f"Investigate gateway retry logic for rider {ride['rider_id']}."
        )

        details = {
            "auth_count": len(auths),
            "first_auth_amount": first_auth["amount"],
            "extra_auth_amounts": [a["amount"] for a in extra_auths],
            "time_gap_seconds": time_gap,
            "amounts_identical": amounts_same,
            "total_captured": total_captured,
            "txn_ids": [a["transaction_id"] for a in auths],
        }

        anomalies.append({
            "ride_id": ride_id,
            "anomaly_type": "duplicate_authorization",
            "impact_category": impact_category,
            "revenue_impact": round(revenue_impact, 2),
            "currency": ride["currency"],
            "country": ride["country"],
            "confidence": confidence,
            "recommendation": recommendation,
            "details": json.dumps(details),
            "detected_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        })

    return anomalies


# ---------------------------------------------------------------------------
# Detector 2: Capture Mismatches (Two-Source Reconciliation)
# ---------------------------------------------------------------------------

def detect_capture_mismatches(sessions):
    """
    Two-source reconciliation: compare capture amount against actual_fare
    from rides table. Only flag if the gap can't be explained by a
    legitimate fare change.

    Threshold: 10% — typical adjustments from route changes, tolls,
    and surge pricing are within 8% of estimated fare.
    """
    anomalies = []

    for ride_id, session in sessions.items():
        ride = session["ride"]
        events = session["events"]

        if ride["status"] != "completed":
            continue

        auths = [e for e in events if e["event_type"] == "authorization" and e["status"] == "approved"]
        captures = [e for e in events if e["event_type"] == "capture" and e["status"] == "approved"]

        if not auths or not captures:
            continue

        auth_amount = auths[0]["amount"]
        capture_amount = captures[0]["amount"]
        actual_fare = ride["actual_fare"]

        if actual_fare == 0:
            continue

        fare_diff_pct = abs(capture_amount - actual_fare) / actual_fare

        # Skip if capture matches actual fare (legitimate)
        if fare_diff_pct <= CAPTURE_MISMATCH_THRESHOLD:
            continue

        if capture_amount > actual_fare:
            direction = "overcharge"
            revenue_impact = capture_amount - actual_fare
            impact_category = "money_at_risk"
        else:
            direction = "undercharge"
            revenue_impact = actual_fare - capture_amount
            impact_category = "money_lost"

        # Confidence based on severity
        if fare_diff_pct > 0.50:
            confidence = 95
        elif fare_diff_pct > 0.35:
            confidence = 90
        elif fare_diff_pct > 0.20:
            confidence = 80
        else:
            confidence = 65
        if direction == "overcharge":
            confidence = min(99, confidence + 5)

        recommendation = (
            f"Review ride {ride_id}: "
            f"{'overcharged' if direction == 'overcharge' else 'under-collected'}. "
            f"Actual fare: {actual_fare:.2f} {ride['currency']}, "
            f"captured: {capture_amount:.2f} ({fare_diff_pct:.0%} diff). "
            f"{'Issue partial refund' if direction == 'overcharge' else 'Investigate uncaptured amount'} "
            f"of {revenue_impact:.2f} {ride['currency']}."
        )

        details = {
            "auth_amount": auth_amount,
            "capture_amount": capture_amount,
            "actual_fare": actual_fare,
            "estimated_fare": ride["estimated_fare"],
            "mismatch_pct": round(fare_diff_pct * 100, 1),
            "direction": direction,
            "auth_txn_id": auths[0]["transaction_id"],
            "capture_txn_id": captures[0]["transaction_id"],
        }

        anomalies.append({
            "ride_id": ride_id,
            "anomaly_type": "capture_mismatch",
            "impact_category": impact_category,
            "revenue_impact": round(revenue_impact, 2),
            "currency": ride["currency"],
            "country": ride["country"],
            "confidence": confidence,
            "recommendation": recommendation,
            "details": json.dumps(details),
            "detected_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        })

    return anomalies


# ---------------------------------------------------------------------------
# Detector 3: Ghost Refunds
# ---------------------------------------------------------------------------

def detect_ghost_refunds(sessions):
    """
    Detect refunds on completed rides with no dispute/cancellation record.
    Cross-references rides, transactions, and disputes tables.
    """
    anomalies = []

    for ride_id, session in sessions.items():
        ride = session["ride"]
        events = session["events"]

        if ride["status"] != "completed" or session["has_dispute"]:
            continue

        refunds = [e for e in events if e["event_type"] == "refund" and e["status"] == "approved"]
        captures = [e for e in events if e["event_type"] == "capture" and e["status"] == "approved"]

        if not refunds:
            continue

        for refund in refunds:
            refund_amount = refund["amount"]
            capture_amount = captures[0]["amount"] if captures else 0

            confidence = 90

            has_null_ref = not refund.get("reference_txn_id") or pd.isna(refund.get("reference_txn_id", None))
            if has_null_ref:
                confidence = max(confidence, 95)

            if capture_amount > 0 and refund_amount > capture_amount:
                confidence = max(confidence, 98)

            days_after = None
            if captures:
                capture_ts = datetime.fromisoformat(captures[0]["timestamp"])
                refund_ts = datetime.fromisoformat(refund["timestamp"])
                days_after = (refund_ts - capture_ts).total_seconds() / 86400
                if days_after > 5:
                    confidence = min(99, confidence + 5)

            recommendation = (
                f"Investigate refund {refund['transaction_id']} of "
                f"{refund_amount:.2f} {ride['currency']} on ride {ride_id}: "
                f"status is 'completed' with no dispute on record. "
                f"{'No reference transaction. ' if has_null_ref else ''}"
                f"Escalate to fraud team."
            )

            details = {
                "refund_amount": refund_amount,
                "capture_amount": capture_amount,
                "refund_txn_id": refund["transaction_id"],
                "reference_txn_id": refund.get("reference_txn_id"),
                "days_after_capture": round(days_after, 1) if days_after else None,
                "ride_status": ride["status"],
                "has_dispute_record": False,
                "refund_exceeds_capture": refund_amount > capture_amount if capture_amount > 0 else None,
            }

            anomalies.append({
                "ride_id": ride_id,
                "anomaly_type": "ghost_refund",
                "impact_category": "money_lost",
                "revenue_impact": round(refund_amount, 2),
                "currency": ride["currency"],
                "country": ride["country"],
                "confidence": confidence,
                "recommendation": recommendation,
                "details": json.dumps(details),
                "detected_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            })

    return anomalies


# ---------------------------------------------------------------------------
# Detector 4: Currency Conversion Discrepancies
# ---------------------------------------------------------------------------

def detect_currency_discrepancies(sessions, exchange_rates):
    """
    Detect captures where amount_usd doesn't match the expected USD conversion
    using the day's exchange rate.

    Thresholds are currency-specific and justified by volatility:
    - MXN: 2% (low volatility, ~0.3%/day)
    - COP: 3% (medium volatility, ~0.5%/day)
    - BRL: 3% (high volatility ~0.6%/day, but auth-to-capture gap
      in ride-hailing is typically <1 hour, not a full trading day)

    These thresholds are tighter than raw daily volatility because
    ride-hailing captures happen within minutes to hours of authorization.
    """
    anomalies = []
    rate_lookup = exchange_rates.set_index(["date", "currency"])["rate_to_usd"]

    for ride_id, session in sessions.items():
        ride = session["ride"]
        events = session["events"]

        captures = [e for e in events if e["event_type"] == "capture" and e["status"] == "approved"]
        if not captures:
            continue

        capture = captures[0]
        if "amount_usd" not in capture or pd.isna(capture.get("amount_usd")):
            continue

        currency = capture["currency"]
        capture_date = capture["timestamp"][:10]
        reported_usd = capture["amount_usd"]

        # Get the expected rate for that day
        try:
            expected_rate = rate_lookup.loc[(capture_date, currency)]
        except KeyError:
            continue

        expected_usd = capture["amount"] / expected_rate
        tolerance = FX_TOLERANCE.get(currency, 0.03)

        if expected_usd == 0:
            continue

        discrepancy_pct = abs(reported_usd - expected_usd) / expected_usd

        if discrepancy_pct < tolerance:
            continue

        revenue_impact = abs(reported_usd - expected_usd)

        # Confidence based on how far beyond tolerance
        excess = discrepancy_pct - tolerance
        if excess > 0.15:
            confidence = 95
        elif excess > 0.08:
            confidence = 85
        elif excess > 0.03:
            confidence = 75
        else:
            confidence = 65

        # BRL is more volatile, so slightly lower confidence for small diffs
        if currency == "BRL" and confidence < 80:
            confidence = max(60, confidence - 5)

        if reported_usd > expected_usd:
            direction = "overstated"
            impact_category = "money_at_risk"
        else:
            direction = "understated"
            impact_category = "money_lost"

        recommendation = (
            f"Currency discrepancy on ride {ride_id}: "
            f"captured {capture['amount']:.2f} {currency}, "
            f"reported as ${reported_usd:.2f} USD but expected "
            f"${expected_usd:.2f} USD ({discrepancy_pct:.1%} diff). "
            f"Review processor FX rate configuration for {currency} transactions."
        )

        details = {
            "local_amount": capture["amount"],
            "reported_usd": reported_usd,
            "expected_usd": round(expected_usd, 2),
            "exchange_rate_used": round(capture["amount"] / reported_usd, 4) if reported_usd > 0 else None,
            "expected_rate": round(expected_rate, 4),
            "discrepancy_pct": round(discrepancy_pct * 100, 1),
            "direction": direction,
            "tolerance_threshold": f"{tolerance:.1%}",
            "capture_txn_id": capture["transaction_id"],
        }

        anomalies.append({
            "ride_id": ride_id,
            "anomaly_type": "currency_discrepancy",
            "impact_category": impact_category,
            "revenue_impact": round(revenue_impact, 2),
            "currency": currency,
            "country": ride["country"],
            "confidence": confidence,
            "recommendation": recommendation,
            "details": json.dumps(details),
            "detected_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        })

    return anomalies


# ---------------------------------------------------------------------------
# Detector 5: Abandoned Authorizations
# ---------------------------------------------------------------------------

def detect_abandoned_authorizations(sessions):
    """
    Detect authorizations on completed rides that were never captured or voided.
    Context matters: cancelled rides with only an auth are normal.
    Completed rides with only an auth = revenue lost.
    """
    anomalies = []

    for ride_id, session in sessions.items():
        ride = session["ride"]
        events = session["events"]

        auths = [e for e in events if e["event_type"] == "authorization" and e["status"] == "approved"]
        captures = [e for e in events if e["event_type"] == "capture"]
        voids = [e for e in events if e["event_type"] == "void"]

        if not auths:
            continue

        # Only flag if ride was completed but payment was never captured
        if captures or voids:
            continue

        if ride["status"] == "completed":
            impact_category = "money_lost"
            revenue_impact = ride["actual_fare"]
            confidence = 95

            auth_ts = datetime.fromisoformat(auths[0]["timestamp"])
            age_days = (datetime.now() - auth_ts).days
            if age_days > 7:
                confidence = min(99, confidence + 3)

            recommendation = (
                f"Abandoned authorization on completed ride {ride_id}: "
                f"authorized {auths[0]['amount']:.2f} {ride['currency']} "
                f"but never captured. Ride fare was {ride['actual_fare']:.2f} {ride['currency']}. "
                f"Attempt late capture or investigate payment processing failure."
            )
        elif ride["status"] == "cancelled":
            # Cancelled ride with dangling auth — funds held unnecessarily
            impact_category = "money_at_risk"
            revenue_impact = auths[0]["amount"]
            confidence = 70

            recommendation = (
                f"Dangling authorization on cancelled ride {ride_id}: "
                f"{auths[0]['amount']:.2f} {ride['currency']} authorized but never voided. "
                f"Release held funds by issuing a void."
            )
        else:
            continue

        details = {
            "auth_amount": auths[0]["amount"],
            "actual_fare": ride["actual_fare"],
            "ride_status": ride["status"],
            "auth_txn_id": auths[0]["transaction_id"],
            "auth_timestamp": auths[0]["timestamp"],
            "has_capture": False,
            "has_void": False,
        }

        anomalies.append({
            "ride_id": ride_id,
            "anomaly_type": "abandoned_authorization",
            "impact_category": impact_category,
            "revenue_impact": round(revenue_impact, 2),
            "currency": ride["currency"],
            "country": ride["country"],
            "confidence": confidence,
            "recommendation": recommendation,
            "details": json.dumps(details),
            "detected_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        })

    return anomalies


# ---------------------------------------------------------------------------
# Currency Conversion Helper
# ---------------------------------------------------------------------------

def convert_to_usd(anomalies_df, exchange_rates):
    """Convert revenue_impact to USD using median exchange rate per currency."""
    if anomalies_df.empty:
        anomalies_df["revenue_impact_usd"] = []
        return anomalies_df

    exchange_rates["date"] = pd.to_datetime(exchange_rates["date"])
    median_rates = exchange_rates.groupby("currency")["rate_to_usd"].median()

    def _convert(row):
        # Currency discrepancies already have impact in USD
        if row["anomaly_type"] == "currency_discrepancy":
            return row["revenue_impact"]
        rate = median_rates.get(row["currency"], 1.0)
        return round(row["revenue_impact"] / rate, 2) if rate > 0 else 0.0

    anomalies_df["revenue_impact_usd"] = anomalies_df.apply(_convert, axis=1)
    return anomalies_df


# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline():
    """Execute the full anomaly detection pipeline."""
    print("=" * 60)
    print("Apollo Rides - Anomaly Detection Pipeline")
    print("=" * 60)

    # Step 1: Load
    print("\n[1/6] Loading data...")
    rides, transactions, disputes, exchange_rates = load_data()
    print(f"      {len(rides)} rides, {len(transactions)} transactions, "
          f"{len(disputes)} dispute/cancellation records")

    # Step 2: Validate
    print("[2/6] Validating data quality...")
    rides, transactions, disputes, validation_report = validate_data(rides, transactions, disputes)
    if validation_report["issues"]:
        for issue in validation_report["issues"]:
            logger.warning(f"  {issue}")
        print(f"      Quarantined {validation_report['quarantined_rows']} rows")
    else:
        print("      No data quality issues found")

    # Step 3: Reconstruct sessions
    print("[3/6] Reconstructing ride payment sessions...")
    sessions = reconstruct_sessions(rides, transactions, disputes)
    lifecycles = {}
    for s in sessions.values():
        lc = s["lifecycle"]
        lifecycles[lc] = lifecycles.get(lc, 0) + 1
    print(f"      {len(sessions)} sessions: {lifecycles}")

    # Step 4: Run all 5 detectors
    print("[4/6] Running anomaly detectors...")
    all_anomalies = []

    detectors = [
        ("Duplicate authorizations", detect_duplicate_authorizations, [sessions]),
        ("Capture mismatches", detect_capture_mismatches, [sessions]),
        ("Ghost refunds", detect_ghost_refunds, [sessions]),
        ("Currency discrepancies", detect_currency_discrepancies, [sessions, exchange_rates]),
        ("Abandoned authorizations", detect_abandoned_authorizations, [sessions]),
    ]

    for name, detector, args in detectors:
        results = detector(*args)
        all_anomalies.extend(results)
        print(f"      {name}: {len(results)} found")

    if not all_anomalies:
        print("\n      No anomalies detected.")
        return pd.DataFrame()

    anomalies_df = pd.DataFrame(all_anomalies)

    # Step 5: Convert to USD
    print("[5/6] Converting revenue impact to USD...")
    anomalies_df = convert_to_usd(anomalies_df, exchange_rates)

    # Step 6: Save
    print("[6/6] Saving results...")
    anomalies_df = anomalies_df.sort_values("confidence", ascending=False)
    anomalies_df.to_csv(f"{DATA_DIR}/anomalies.csv", index=False)

    validation_df = pd.DataFrame({
        "issue": validation_report["issues"] if validation_report["issues"] else ["No issues"],
        "quarantined_rows": [validation_report["quarantined_rows"]] * max(1, len(validation_report["issues"])),
    })
    validation_df.to_csv(f"{DATA_DIR}/validation_report.csv", index=False)

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nTotal anomalies: {len(anomalies_df)}")

    print(f"\nBy type:")
    for atype in anomalies_df["anomaly_type"].unique():
        type_df = anomalies_df[anomalies_df["anomaly_type"] == atype]
        total_usd = type_df["revenue_impact_usd"].sum()
        print(f"  {atype}: {len(type_df)} anomalies, ${total_usd:,.2f} USD")

    print(f"\nBy impact category:")
    for cat in ["money_lost", "money_at_risk"]:
        group = anomalies_df[anomalies_df["impact_category"] == cat]
        if len(group) > 0:
            print(f"  {cat}: {len(group)} anomalies, ${group['revenue_impact_usd'].sum():,.2f} USD")

    print(f"\nBy country:")
    for country in sorted(anomalies_df["country"].unique()):
        group = anomalies_df[anomalies_df["country"] == country]
        print(f"  {country}: {len(group)} anomalies, ${group['revenue_impact_usd'].sum():,.2f} USD")

    print(f"\nAverage confidence: {anomalies_df['confidence'].mean():.1f}%")
    print(f"High confidence (>80%): {(anomalies_df['confidence'] > 80).sum()}")
    print(f"\nSaved to {DATA_DIR}/anomalies.csv")
    print("=" * 60)

    return anomalies_df


if __name__ == "__main__":
    run_pipeline()
