"""
Apollo Rides - Test Data Generator

Generates realistic payment transaction data with injected anomalies
for testing the anomaly detection pipeline.

Data sources generated:
  - rides.csv: Business records of completed/cancelled/disputed rides
  - transactions.csv: Payment event logs (auth, capture, void, refund)
  - disputes_cancellations.csv: Records proving legitimate refunds/voids
  - exchange_rates.csv: Daily currency conversion rates
"""

import os
import random
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Configuration
DATA_DIR = "data"
NUM_RIDES = 500
NUM_RIDERS = 80
NUM_DRIVERS = 40
START_DATE = datetime(2025, 12, 1)
END_DATE = datetime(2026, 2, 23)

COUNTRIES = {
    "MX": {"currency": "MXN", "weight": 0.40, "fare_range": (45, 650)},
    "CO": {"currency": "COP", "weight": 0.35, "fare_range": (5000, 85000)},
    "BR": {"currency": "BRL", "weight": 0.25, "fare_range": (12, 180)},
}

# Exchange rate baselines (currency per 1 USD)
RATE_BASELINES = {"MXN": 17.2, "COP": 4150.0, "BRL": 5.1}

# Anomaly budget (~30% of rides)
ANOMALY_COUNTS = {
    "duplicate_auth": 30,
    "capture_mismatch": 35,
    "ghost_refund": 30,
    "currency_discrepancy": 30,
    "abandoned_auth": 25,
}

# Counters
_txn_counter = 0
_record_counter = 0


def next_txn_id():
    global _txn_counter
    _txn_counter += 1
    return f"TXN-{_txn_counter:05d}"


def next_record_id():
    global _record_counter
    _record_counter += 1
    return f"REC-{_record_counter:05d}"


def generate_exchange_rates():
    """Generate daily exchange rates with realistic random walk."""
    dates = pd.date_range(START_DATE.date(), END_DATE.date(), freq="D")
    rows = []
    rates = {c: v for c, v in RATE_BASELINES.items()}

    for d in dates:
        for currency, baseline in RATE_BASELINES.items():
            # Random walk: daily volatility ~0.3-0.8% depending on currency
            volatility = {"MXN": 0.003, "COP": 0.005, "BRL": 0.006}[currency]
            rates[currency] *= 1 + np.random.normal(0, volatility)
            # Keep within realistic bounds
            rates[currency] = max(baseline * 0.9, min(baseline * 1.1, rates[currency]))
            rows.append({
                "date": d.strftime("%Y-%m-%d"),
                "currency": currency,
                "rate_to_usd": round(rates[currency], 4),
            })

    return pd.DataFrame(rows)


def pick_country():
    """Weighted random country selection."""
    countries = list(COUNTRIES.keys())
    weights = [COUNTRIES[c]["weight"] for c in countries]
    return random.choices(countries, weights=weights, k=1)[0]


def generate_fare(country):
    """Generate realistic fare using log-normal distribution."""
    lo, hi = COUNTRIES[country]["fare_range"]
    mid = (lo + hi) / 2
    sigma = 0.5
    fare = np.random.lognormal(np.log(mid), sigma)
    return round(max(lo, min(hi, fare)), 2)


def generate_rides():
    """Generate ride records with status distribution."""
    riders = [f"RIDER-{i:04d}" for i in range(1, NUM_RIDERS + 1)]
    drivers = [f"DRIVER-{i:04d}" for i in range(1, NUM_DRIVERS + 1)]

    # Power-law: some riders are more frequent
    rider_weights = np.random.pareto(1.5, NUM_RIDERS) + 1
    rider_weights /= rider_weights.sum()

    total_seconds = int((END_DATE - START_DATE).total_seconds())
    rides = []

    for i in range(1, NUM_RIDES + 1):
        country = pick_country()
        currency = COUNTRIES[country]["currency"]
        estimated_fare = generate_fare(country)
        # Actual fare: small natural variance (-8% to +5% of estimated)
        fare_adjustment = np.random.uniform(-0.08, 0.05)
        actual_fare = round(estimated_fare * (1 + fare_adjustment), 2)

        ts = START_DATE + timedelta(seconds=random.randint(0, total_seconds))

        # Status distribution: 85% completed, 10% cancelled, 5% disputed
        status_roll = random.random()
        if status_roll < 0.85:
            status = "completed"
        elif status_roll < 0.95:
            status = "cancelled"
        else:
            status = "disputed"

        rides.append({
            "ride_id": f"RIDE-{i:05d}",
            "rider_id": np.random.choice(riders, p=rider_weights),
            "driver_id": random.choice(drivers),
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S"),
            "country": country,
            "currency": currency,
            "estimated_fare": estimated_fare,
            "actual_fare": actual_fare,
            "status": status,
        })

    return pd.DataFrame(rides)


def generate_clean_transactions(ride):
    """Generate normal transaction flow based on ride status."""
    txns = []
    ride_ts = datetime.fromisoformat(ride["timestamp"])

    if ride["status"] == "completed":
        # auth -> capture
        auth_id = next_txn_id()
        txns.append({
            "transaction_id": auth_id,
            "ride_id": ride["ride_id"],
            "event_type": "authorization",
            "amount": ride["estimated_fare"],
            "currency": ride["currency"],
            "timestamp": (ride_ts + timedelta(minutes=random.randint(0, 2))).strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "approved",
            "reference_txn_id": None,
        })
        txns.append({
            "transaction_id": next_txn_id(),
            "ride_id": ride["ride_id"],
            "event_type": "capture",
            "amount": ride["actual_fare"],
            "currency": ride["currency"],
            "timestamp": (ride_ts + timedelta(minutes=random.randint(15, 45))).strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "approved",
            "reference_txn_id": auth_id,
        })

    elif ride["status"] == "cancelled":
        # auth -> void
        auth_id = next_txn_id()
        txns.append({
            "transaction_id": auth_id,
            "ride_id": ride["ride_id"],
            "event_type": "authorization",
            "amount": ride["estimated_fare"],
            "currency": ride["currency"],
            "timestamp": (ride_ts + timedelta(minutes=random.randint(0, 2))).strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "approved",
            "reference_txn_id": None,
        })
        txns.append({
            "transaction_id": next_txn_id(),
            "ride_id": ride["ride_id"],
            "event_type": "void",
            "amount": ride["estimated_fare"],
            "currency": ride["currency"],
            "timestamp": (ride_ts + timedelta(minutes=random.randint(5, 30))).strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "approved",
            "reference_txn_id": auth_id,
        })

    elif ride["status"] == "disputed":
        # auth -> capture -> refund (legitimate, with dispute record)
        auth_id = next_txn_id()
        capture_id = next_txn_id()
        txns.append({
            "transaction_id": auth_id,
            "ride_id": ride["ride_id"],
            "event_type": "authorization",
            "amount": ride["estimated_fare"],
            "currency": ride["currency"],
            "timestamp": (ride_ts + timedelta(minutes=random.randint(0, 2))).strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "approved",
            "reference_txn_id": None,
        })
        txns.append({
            "transaction_id": capture_id,
            "ride_id": ride["ride_id"],
            "event_type": "capture",
            "amount": ride["actual_fare"],
            "currency": ride["currency"],
            "timestamp": (ride_ts + timedelta(minutes=random.randint(15, 45))).strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "approved",
            "reference_txn_id": auth_id,
        })
        txns.append({
            "transaction_id": next_txn_id(),
            "ride_id": ride["ride_id"],
            "event_type": "refund",
            "amount": ride["actual_fare"],
            "currency": ride["currency"],
            "timestamp": (ride_ts + timedelta(days=random.randint(1, 5))).strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "approved",
            "reference_txn_id": capture_id,
        })

    return txns


def inject_duplicate_auth(ride, txns):
    """
    Inject a duplicate authorization with severity spectrum:
    - Subtle: amount differs by 1-3%, 5s gap
    - Moderate: same amount, 30s gap
    - Obvious: 3+ auths
    """
    auth_txn = next((t for t in txns if t["event_type"] == "authorization"), None)
    if not auth_txn:
        return txns

    auth_ts = datetime.fromisoformat(auth_txn["timestamp"])
    severity = random.choices(["subtle", "moderate", "obvious"], weights=[0.4, 0.4, 0.2], k=1)[0]

    if severity == "subtle":
        # Slightly different amount, very close timestamp
        amount_adj = auth_txn["amount"] * np.random.uniform(0.97, 1.03)
        dup_ts = auth_ts + timedelta(seconds=random.randint(2, 8))
        txns.append({
            "transaction_id": next_txn_id(),
            "ride_id": ride["ride_id"],
            "event_type": "authorization",
            "amount": round(amount_adj, 2),
            "currency": ride["currency"],
            "timestamp": dup_ts.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "approved",
            "reference_txn_id": None,
        })
    elif severity == "moderate":
        # Same amount, moderate gap
        dup_ts = auth_ts + timedelta(seconds=random.randint(15, 45))
        txns.append({
            "transaction_id": next_txn_id(),
            "ride_id": ride["ride_id"],
            "event_type": "authorization",
            "amount": auth_txn["amount"],
            "currency": ride["currency"],
            "timestamp": dup_ts.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "approved",
            "reference_txn_id": None,
        })
    else:
        # 3 auths total
        for offset in [5, 15]:
            dup_ts = auth_ts + timedelta(seconds=offset)
            txns.append({
                "transaction_id": next_txn_id(),
                "ride_id": ride["ride_id"],
                "event_type": "authorization",
                "amount": auth_txn["amount"],
                "currency": ride["currency"],
                "timestamp": dup_ts.strftime("%Y-%m-%dT%H:%M:%S"),
                "status": "approved",
                "reference_txn_id": None,
            })

    return txns


def inject_capture_mismatch(ride, txns):
    """
    Inject capture amount mismatch with severity spectrum:
    - Subtle: 12-18% difference
    - Moderate: 25-40% difference
    - Obvious: >50% difference
    Mismatch is relative to ACTUAL fare (two-source reconciliation).
    """
    capture_txn = next((t for t in txns if t["event_type"] == "capture"), None)
    if not capture_txn:
        return txns

    severity = random.choices(["subtle", "moderate", "obvious"], weights=[0.4, 0.35, 0.25], k=1)[0]
    direction = random.choice(["overcharge", "undercharge"])

    if severity == "subtle":
        pct = np.random.uniform(0.12, 0.18)
    elif severity == "moderate":
        pct = np.random.uniform(0.25, 0.40)
    else:
        pct = np.random.uniform(0.50, 0.70)

    if direction == "overcharge":
        capture_txn["amount"] = round(ride["actual_fare"] * (1 + pct), 2)
    else:
        capture_txn["amount"] = round(ride["actual_fare"] * (1 - pct), 2)

    return txns


def inject_ghost_refund(ride, txns):
    """
    Inject a refund on a completed ride with NO dispute/cancellation record.
    Severity spectrum:
    - Subtle: partial refund (30-60%), 1-2 days later
    - Moderate: full refund, 3-5 days later
    - Obvious: refund exceeds capture amount
    """
    capture_txn = next((t for t in txns if t["event_type"] == "capture"), None)
    if not capture_txn:
        return txns

    capture_ts = datetime.fromisoformat(capture_txn["timestamp"])
    severity = random.choices(["subtle", "moderate", "obvious"], weights=[0.4, 0.35, 0.25], k=1)[0]

    if severity == "subtle":
        refund_amount = round(capture_txn["amount"] * np.random.uniform(0.30, 0.60), 2)
        refund_ts = capture_ts + timedelta(days=random.randint(1, 2), hours=random.randint(1, 12))
    elif severity == "moderate":
        refund_amount = capture_txn["amount"]
        refund_ts = capture_ts + timedelta(days=random.randint(3, 5), hours=random.randint(1, 12))
    else:
        refund_amount = round(capture_txn["amount"] * np.random.uniform(1.05, 1.30), 2)
        refund_ts = capture_ts + timedelta(days=random.randint(5, 10), hours=random.randint(1, 12))

    # Randomly, some ghost refunds have null reference (extra suspicious)
    ref_id = capture_txn["transaction_id"] if random.random() > 0.25 else None

    txns.append({
        "transaction_id": next_txn_id(),
        "ride_id": ride["ride_id"],
        "event_type": "refund",
        "amount": refund_amount,
        "currency": ride["currency"],
        "timestamp": refund_ts.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "approved",
        "reference_txn_id": ref_id,
    })

    return txns


def inject_currency_discrepancy(ride, txns, exchange_rates_df):
    """
    Inject currency conversion discrepancy: the capture amount in local currency
    doesn't match what it should be when converted via the day's exchange rate.
    We add an amount_usd field that uses a skewed rate.
    """
    capture_txn = next((t for t in txns if t["event_type"] == "capture"), None)
    if not capture_txn:
        return txns

    severity = random.choices(["subtle", "moderate", "obvious"], weights=[0.4, 0.35, 0.25], k=1)[0]

    if severity == "subtle":
        skew = np.random.uniform(0.03, 0.06)
    elif severity == "moderate":
        skew = np.random.uniform(0.08, 0.15)
    else:
        skew = np.random.uniform(0.18, 0.30)

    direction = random.choice([-1, 1])
    capture_txn["fx_rate_applied"] = "skewed"
    capture_txn["_fx_skew"] = round(skew * direction, 4)

    return txns


def inject_abandoned_auth(ride, txns):
    """
    Inject an abandoned authorization: auth was approved but never captured or voided.
    Remove capture/void events, leaving the auth dangling.
    """
    txns = [t for t in txns if t["event_type"] == "authorization"]
    return txns


def generate_disputes_cancellations(rides_df):
    """
    Generate dispute/cancellation records ONLY for rides with
    status 'disputed' or 'cancelled'. This proves ghost refund detection works:
    completed rides with refunds will have NO matching record here.
    """
    records = []
    reasons_dispute = [
        "Driver took wrong route",
        "Ride quality issue",
        "Incorrect fare charged",
        "Driver no-show",
        "Safety concern reported",
    ]
    reasons_cancel = [
        "Rider cancelled before pickup",
        "Driver cancelled",
        "No driver available",
        "Rider changed plans",
        "Duplicate booking",
    ]

    for _, ride in rides_df.iterrows():
        ride_ts = datetime.fromisoformat(ride["timestamp"])

        if ride["status"] == "disputed":
            records.append({
                "record_id": next_record_id(),
                "ride_id": ride["ride_id"],
                "type": "dispute",
                "reason": random.choice(reasons_dispute),
                "timestamp": (ride_ts + timedelta(hours=random.randint(1, 48))).strftime("%Y-%m-%dT%H:%M:%S"),
            })
        elif ride["status"] == "cancelled":
            records.append({
                "record_id": next_record_id(),
                "ride_id": ride["ride_id"],
                "type": "cancellation",
                "reason": random.choice(reasons_cancel),
                "timestamp": (ride_ts + timedelta(minutes=random.randint(1, 15))).strftime("%Y-%m-%dT%H:%M:%S"),
            })

    return pd.DataFrame(records)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 60)
    print("Apollo Rides - Test Data Generator")
    print("=" * 60)

    # Step 1: Exchange rates
    exchange_rates = generate_exchange_rates()
    exchange_rates.to_csv(f"{DATA_DIR}/exchange_rates.csv", index=False)
    print(f"[OK] Exchange rates: {len(exchange_rates)} rows")

    # Step 2: Rides
    rides_df = generate_rides()

    # Step 3: Assign anomalies to completed rides only
    completed_mask = rides_df["status"] == "completed"
    completed_indices = rides_df[completed_mask].index.tolist()
    random.shuffle(completed_indices)

    anomaly_assignments = {}
    idx = 0
    for anomaly_type, count in ANOMALY_COUNTS.items():
        for _ in range(count):
            if idx < len(completed_indices):
                anomaly_assignments[completed_indices[idx]] = anomaly_type
                idx += 1

    # Step 4: Generate transactions with anomaly injection
    all_txns = []
    injected_counts = {k: 0 for k in ANOMALY_COUNTS}

    for i, ride in rides_df.iterrows():
        txns = generate_clean_transactions(ride.to_dict())

        if i in anomaly_assignments:
            atype = anomaly_assignments[i]
            if atype == "duplicate_auth":
                txns = inject_duplicate_auth(ride.to_dict(), txns)
            elif atype == "capture_mismatch":
                txns = inject_capture_mismatch(ride.to_dict(), txns)
            elif atype == "ghost_refund":
                txns = inject_ghost_refund(ride.to_dict(), txns)
            elif atype == "currency_discrepancy":
                txns = inject_currency_discrepancy(ride.to_dict(), txns, exchange_rates)
            elif atype == "abandoned_auth":
                txns = inject_abandoned_auth(ride.to_dict(), txns)
            injected_counts[atype] += 1

        all_txns.extend(txns)

    transactions_df = pd.DataFrame(all_txns)

    # Add USD conversion using exchange rates
    rate_lookup = exchange_rates.set_index(["date", "currency"])["rate_to_usd"]

    def convert_to_usd(row):
        txn_date = row["timestamp"][:10]
        currency = row["currency"]
        skew = row.get("_fx_skew", 0) or 0
        try:
            rate = rate_lookup.loc[(txn_date, currency)]
        except KeyError:
            rate = RATE_BASELINES.get(currency, 1.0)
        effective_rate = rate * (1 + skew)
        return round(row["amount"] / effective_rate, 2)

    if "_fx_skew" not in transactions_df.columns:
        transactions_df["_fx_skew"] = 0
    transactions_df["_fx_skew"] = transactions_df["_fx_skew"].fillna(0)
    transactions_df["amount_usd"] = transactions_df.apply(convert_to_usd, axis=1)
    # Clean up internal columns
    transactions_df.drop(columns=["_fx_skew", "fx_rate_applied"], errors="ignore", inplace=True)

    # Step 5: Disputes & cancellations (only for disputed/cancelled rides)
    disputes_df = generate_disputes_cancellations(rides_df)

    # Save anomaly ground truth for validation
    ground_truth = []
    for idx, atype in anomaly_assignments.items():
        ground_truth.append({
            "ride_id": rides_df.loc[idx, "ride_id"],
            "anomaly_type": atype,
        })
    gt_df = pd.DataFrame(ground_truth)
    gt_df.to_csv(f"{DATA_DIR}/ground_truth.csv", index=False)

    # Save everything
    rides_df.to_csv(f"{DATA_DIR}/rides.csv", index=False)
    transactions_df.to_csv(f"{DATA_DIR}/transactions.csv", index=False)
    disputes_df.to_csv(f"{DATA_DIR}/disputes_cancellations.csv", index=False)

    # Summary
    print(f"[OK] Rides: {len(rides_df)} total")
    print(f"     - Completed: {(rides_df['status'] == 'completed').sum()}")
    print(f"     - Cancelled: {(rides_df['status'] == 'cancelled').sum()}")
    print(f"     - Disputed: {(rides_df['status'] == 'disputed').sum()}")
    print(f"     - Countries: {rides_df['country'].value_counts().to_dict()}")
    print(f"[OK] Transactions: {len(transactions_df)} events")
    print(f"     - Event types: {transactions_df['event_type'].value_counts().to_dict()}")
    print(f"[OK] Disputes/Cancellations: {len(disputes_df)} records")
    print(f"[OK] Injected anomalies: {injected_counts}")
    print(f"\nAll files saved to {DATA_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
