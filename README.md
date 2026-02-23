# Apollo Rides — Transaction Anomaly Detector

A post-transaction analytical system that identifies revenue leaks in Apollo Rides' payment data across Mexico, Colombia, and Brazil. Built for the Yuno engineering challenge.

## Architecture

```
generate_data.py ──→ data/rides.csv                  ┐
                     data/transactions.csv            ├──→ detect_anomalies.py ──→ data/anomalies.csv ──→ dashboard.py
                     data/disputes_cancellations.csv  │
                     data/exchange_rates.csv          ┘
```

**Three-stage pipeline:**
1. **Data Generation** — Creates realistic test data with ~500 rides, 4 related tables, and injected anomalies across a severity spectrum
2. **Anomaly Detection** — Session-based pipeline that reconstructs ride payment lifecycles, validates data quality, and runs 5 detectors with confidence scoring
3. **Dashboard** — Narrative Streamlit app that tells the revenue leak story to non-technical stakeholders

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate test data (~500 rides, ~1000+ transaction events)
python generate_data.py

# Run anomaly detection pipeline
python detect_anomalies.py

# Launch dashboard
streamlit run dashboard.py
```

## Data Model — Two-Source Reconciliation

Anomalies are found in the **gap** between what the business says happened and what the payment system recorded.

| Table | Purpose | Key Fields |
|-------|---------|------------|
| `rides.csv` | Business truth — what rides occurred | ride_id, actual_fare, estimated_fare, status, country |
| `transactions.csv` | Payment truth — what the system processed | transaction_id, ride_id, event_type, amount, amount_usd |
| `disputes_cancellations.csv` | Justification records for refunds/voids | ride_id, type (dispute/cancellation), reason |
| `exchange_rates.csv` | Daily FX rates for MXN, COP, BRL to USD | date, currency, rate_to_usd |

## Detection Logic

### 1. Duplicate Authorizations
**What:** Multiple approved authorizations for the same ride (payment gateway retry/double-tap).

**How:** Group transactions by ride_id, filter authorizations with status=approved, flag rides with count > 1.

**Revenue impact (type-specific):**
- Extra auth was also captured → **money_lost** (rider was double-charged)
- Extra auth was NOT captured → **money_at_risk** (funds held on rider's card)

**Confidence:** Same amount + <60s gap = 95% | Different amounts = 85% | 3+ auths = 98%

### 2. Capture Mismatches (Two-Source Reconciliation)
**What:** The captured amount doesn't match the actual ride fare.

**How:** Compare `capture.amount` against `rides.actual_fare` (NOT against the authorization amount). This cross-references two data sources — a mismatch that can't be explained by a legitimate fare change is a real anomaly.

**Threshold:** 10% — justified because typical fare adjustments from route changes, tolls, and surge pricing are within 8%.

**Revenue impact:** `actual_fare - captured_amount` for undercharges (money_lost), inverse for overcharges (money_at_risk).

**Confidence:** 10-20% diff = 65% | 20-35% = 80% | >35% = 90-95%

### 3. Ghost Refunds
**What:** Refunds issued on completed rides with no dispute or cancellation record.

**How:** Find refund events → join with rides (status must be "completed") → join with disputes_cancellations (must have NO matching record). This three-way cross-reference proves the refund was unauthorized.

**Revenue impact:** Full refund amount = **money_lost** (direct revenue leaving the platform).

**Confidence:** No dispute = 90% | Null reference txn = 95% | Refund > capture = 98%

### 4. Currency Conversion Discrepancies
**What:** The USD conversion of a capture doesn't match the expected rate for that day.

**How:** Compare `amount_usd` against `amount / exchange_rate` for the capture date.

**Thresholds (justified by currency volatility):**
- MXN: 2% tolerance (low volatility, ~0.3%/day)
- COP: 3% tolerance (medium volatility, ~0.5%/day)
- BRL: 3.5% tolerance (high volatility, ~0.6%/day)

These account for the hours-long gap between authorization and capture during which rates can drift.

### 5. Abandoned Authorizations
**What:** Rides that were authorized but never captured or voided.

**How:** Find rides where the only transaction event is an authorization (no capture, no void). Context matters: completed rides = revenue lost, cancelled rides with no void = unnecessary funds hold.

**Revenue impact:** Completed ride → `actual_fare` = **money_lost** | Cancelled ride → auth amount = **money_at_risk**

## Test Data Design

- **500 rides** across Dec 2025 — Feb 2026
- **3 currencies:** MXN (Mexico 40%), COP (Colombia 35%), BRL (Brazil 25%)
- **~70% clean, ~30% anomalous** with ground truth saved to `data/ground_truth.csv`
- **Severity spectrum:** Each anomaly type has subtle (borderline), moderate, and obvious variants
- **Legitimate edge cases** that should NOT be flagged: fare adjustments, disputed rides with justified refunds, cancelled rides with proper voids

## Key Findings (Sample Run)

| Metric | Value |
|--------|-------|
| Total anomalies | 148 |
| Revenue lost (confirmed) | ~$1,005 USD |
| Revenue at risk | ~$800 USD |
| Most impactful type | Duplicate authorizations (~$700 USD) |
| Most affected country | Mexico (64 anomalies, ~$982 USD) |
| Average confidence | 89.1% |
| High confidence (>80%) | 121 anomalies |

## Stretch Goals Implemented

- **Confidence scoring (0-100%)** — Each anomaly has a calibrated confidence score based on type-specific heuristics
- **Actionable recommendations** — Every anomaly includes a specific recommendation (e.g., "Issue partial refund of X", "Escalate to fraud team")
- **Root cause hypotheses** — Dashboard insights section generates data-driven hypotheses about where problems originate
- **Impact categorization** — Revenue classified as "money_lost" vs "money_at_risk" with type-specific formulas

## Technology Choices

| Choice | Rationale |
|--------|-----------|
| **Python + pandas** | Fastest path for data processing, widely understood |
| **Streamlit** | Interactive dashboard with minimal code, professional appearance |
| **Plotly** | Rich interactive charts with hover details, consistent styling |
| **Session-based detection** | Reconstructing ride lifecycles catches patterns that row-by-row analysis misses |
| **No database** | CSV-based for simplicity — real production system would use a data warehouse |

## What I'd Build Next (4 more hours)

1. **Automated alerting** — Slack/email notifications when anomaly rate exceeds a threshold
2. **ML-based anomaly detection** — Isolation Forest or DBSCAN to find novel patterns beyond the 5 rule-based detectors
3. **Provider/processor segmentation** — Break down anomalies by payment processor to identify misconfigured integrations
4. **Historical comparison** — Week-over-week trend analysis with statistical significance testing
5. **Data pipeline orchestration** — Airflow/Prefect DAG for scheduled runs against a real data warehouse
6. **API endpoint** — REST API for programmatic access to anomaly results
