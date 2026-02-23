"""
Unit tests for the anomaly detection pipeline.

Tests each detector with controlled inputs to verify:
- Anomalies are correctly identified
- Legitimate transactions are NOT flagged (no false positives)
- Revenue impact calculations are type-specific
- Confidence scores follow documented rules
"""

from __future__ import annotations

import json
import unittest
import pandas as pd
from detect_anomalies import (
    detect_duplicate_authorizations,
    detect_capture_mismatches,
    detect_ghost_refunds,
    detect_abandoned_authorizations,
    detect_currency_discrepancies,
    reconstruct_sessions,
    validate_data,
)


def make_ride(ride_id: str = "RIDE-TEST", country: str = "MX",
              currency: str = "MXN", estimated: float = 200.0,
              actual: float = 195.0, status: str = "completed") -> pd.DataFrame:
    """Helper: create a single-ride DataFrame."""
    return pd.DataFrame([{
        "ride_id": ride_id, "rider_id": "RIDER-0001", "driver_id": "DRIVER-0001",
        "timestamp": "2026-01-15T10:00:00", "country": country, "currency": currency,
        "estimated_fare": estimated, "actual_fare": actual, "status": status,
    }])


def make_txns(rows: list[dict]) -> pd.DataFrame:
    """Helper: create transactions DataFrame from list of dicts."""
    return pd.DataFrame(rows)


def make_disputes(rows: list[dict] | None = None) -> pd.DataFrame:
    """Helper: create disputes DataFrame."""
    if rows is None:
        rows = []
    return pd.DataFrame(rows, columns=["record_id", "ride_id", "type", "reason", "timestamp"])


class TestDuplicateAuthorizations(unittest.TestCase):

    def test_detects_duplicate_auth(self):
        """Two approved auths for the same ride should be flagged."""
        rides = make_ride()
        txns = make_txns([
            {"transaction_id": "T1", "ride_id": "RIDE-TEST", "event_type": "authorization",
             "amount": 200.0, "currency": "MXN", "timestamp": "2026-01-15T10:00:00",
             "status": "approved", "reference_txn_id": None},
            {"transaction_id": "T2", "ride_id": "RIDE-TEST", "event_type": "authorization",
             "amount": 200.0, "currency": "MXN", "timestamp": "2026-01-15T10:00:05",
             "status": "approved", "reference_txn_id": None},
            {"transaction_id": "T3", "ride_id": "RIDE-TEST", "event_type": "capture",
             "amount": 195.0, "currency": "MXN", "timestamp": "2026-01-15T10:30:00",
             "status": "approved", "reference_txn_id": "T1"},
        ])
        sessions = reconstruct_sessions(rides, txns, make_disputes())
        anomalies = detect_duplicate_authorizations(sessions)

        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0]["anomaly_type"], "duplicate_authorization")
        self.assertEqual(anomalies[0]["confidence"], 95)  # same amount, <60s

    def test_single_auth_not_flagged(self):
        """A normal ride with one auth should not be flagged."""
        rides = make_ride()
        txns = make_txns([
            {"transaction_id": "T1", "ride_id": "RIDE-TEST", "event_type": "authorization",
             "amount": 200.0, "currency": "MXN", "timestamp": "2026-01-15T10:00:00",
             "status": "approved", "reference_txn_id": None},
            {"transaction_id": "T2", "ride_id": "RIDE-TEST", "event_type": "capture",
             "amount": 195.0, "currency": "MXN", "timestamp": "2026-01-15T10:30:00",
             "status": "approved", "reference_txn_id": "T1"},
        ])
        sessions = reconstruct_sessions(rides, txns, make_disputes())
        anomalies = detect_duplicate_authorizations(sessions)

        self.assertEqual(len(anomalies), 0)

    def test_triple_auth_high_confidence(self):
        """Three auths should get 98% confidence."""
        rides = make_ride()
        txns = make_txns([
            {"transaction_id": f"T{i}", "ride_id": "RIDE-TEST", "event_type": "authorization",
             "amount": 200.0, "currency": "MXN",
             "timestamp": f"2026-01-15T10:00:{i*5:02d}",
             "status": "approved", "reference_txn_id": None}
            for i in range(3)
        ])
        sessions = reconstruct_sessions(rides, txns, make_disputes())
        anomalies = detect_duplicate_authorizations(sessions)

        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0]["confidence"], 98)


class TestCaptureMismatches(unittest.TestCase):

    def test_detects_overcharge(self):
        """Capture significantly above actual fare should be flagged."""
        rides = make_ride(actual=195.0)
        txns = make_txns([
            {"transaction_id": "T1", "ride_id": "RIDE-TEST", "event_type": "authorization",
             "amount": 200.0, "currency": "MXN", "timestamp": "2026-01-15T10:00:00",
             "status": "approved", "reference_txn_id": None},
            {"transaction_id": "T2", "ride_id": "RIDE-TEST", "event_type": "capture",
             "amount": 300.0, "currency": "MXN", "timestamp": "2026-01-15T10:30:00",
             "status": "approved", "reference_txn_id": "T1"},
        ])
        sessions = reconstruct_sessions(rides, txns, make_disputes())
        anomalies = detect_capture_mismatches(sessions)

        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0]["impact_category"], "money_at_risk")
        details = json.loads(anomalies[0]["details"])
        self.assertEqual(details["direction"], "overcharge")

    def test_small_diff_not_flagged(self):
        """Capture within 10% of actual fare should not be flagged (legitimate)."""
        rides = make_ride(actual=195.0)
        txns = make_txns([
            {"transaction_id": "T1", "ride_id": "RIDE-TEST", "event_type": "authorization",
             "amount": 200.0, "currency": "MXN", "timestamp": "2026-01-15T10:00:00",
             "status": "approved", "reference_txn_id": None},
            {"transaction_id": "T2", "ride_id": "RIDE-TEST", "event_type": "capture",
             "amount": 190.0, "currency": "MXN", "timestamp": "2026-01-15T10:30:00",
             "status": "approved", "reference_txn_id": "T1"},
        ])
        sessions = reconstruct_sessions(rides, txns, make_disputes())
        anomalies = detect_capture_mismatches(sessions)

        self.assertEqual(len(anomalies), 0)  # 2.5% diff — below threshold


class TestGhostRefunds(unittest.TestCase):

    def test_detects_ghost_refund(self):
        """Refund on completed ride with no dispute should be flagged."""
        rides = make_ride(status="completed")
        txns = make_txns([
            {"transaction_id": "T1", "ride_id": "RIDE-TEST", "event_type": "authorization",
             "amount": 200.0, "currency": "MXN", "timestamp": "2026-01-15T10:00:00",
             "status": "approved", "reference_txn_id": None},
            {"transaction_id": "T2", "ride_id": "RIDE-TEST", "event_type": "capture",
             "amount": 195.0, "currency": "MXN", "timestamp": "2026-01-15T10:30:00",
             "status": "approved", "reference_txn_id": "T1"},
            {"transaction_id": "T3", "ride_id": "RIDE-TEST", "event_type": "refund",
             "amount": 195.0, "currency": "MXN", "timestamp": "2026-01-18T14:00:00",
             "status": "approved", "reference_txn_id": "T2"},
        ])
        sessions = reconstruct_sessions(rides, txns, make_disputes())
        anomalies = detect_ghost_refunds(sessions)

        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0]["impact_category"], "money_lost")
        self.assertEqual(anomalies[0]["revenue_impact"], 195.0)

    def test_disputed_refund_not_flagged(self):
        """Refund on disputed ride with matching dispute record should NOT be flagged."""
        rides = make_ride(status="disputed")
        txns = make_txns([
            {"transaction_id": "T1", "ride_id": "RIDE-TEST", "event_type": "authorization",
             "amount": 200.0, "currency": "MXN", "timestamp": "2026-01-15T10:00:00",
             "status": "approved", "reference_txn_id": None},
            {"transaction_id": "T2", "ride_id": "RIDE-TEST", "event_type": "capture",
             "amount": 195.0, "currency": "MXN", "timestamp": "2026-01-15T10:30:00",
             "status": "approved", "reference_txn_id": "T1"},
            {"transaction_id": "T3", "ride_id": "RIDE-TEST", "event_type": "refund",
             "amount": 195.0, "currency": "MXN", "timestamp": "2026-01-18T14:00:00",
             "status": "approved", "reference_txn_id": "T2"},
        ])
        disputes = make_disputes([{
            "record_id": "REC-1", "ride_id": "RIDE-TEST",
            "type": "dispute", "reason": "Wrong route",
            "timestamp": "2026-01-16T12:00:00",
        }])
        sessions = reconstruct_sessions(rides, txns, disputes)
        anomalies = detect_ghost_refunds(sessions)

        self.assertEqual(len(anomalies), 0)  # legitimate refund

    def test_null_reference_high_confidence(self):
        """Ghost refund with null reference_txn_id should get >= 95% confidence."""
        rides = make_ride(status="completed")
        txns = make_txns([
            {"transaction_id": "T1", "ride_id": "RIDE-TEST", "event_type": "authorization",
             "amount": 200.0, "currency": "MXN", "timestamp": "2026-01-15T10:00:00",
             "status": "approved", "reference_txn_id": None},
            {"transaction_id": "T2", "ride_id": "RIDE-TEST", "event_type": "capture",
             "amount": 195.0, "currency": "MXN", "timestamp": "2026-01-15T10:30:00",
             "status": "approved", "reference_txn_id": "T1"},
            {"transaction_id": "T3", "ride_id": "RIDE-TEST", "event_type": "refund",
             "amount": 195.0, "currency": "MXN", "timestamp": "2026-01-18T14:00:00",
             "status": "approved", "reference_txn_id": None},
        ])
        sessions = reconstruct_sessions(rides, txns, make_disputes())
        anomalies = detect_ghost_refunds(sessions)

        self.assertEqual(len(anomalies), 1)
        self.assertGreaterEqual(anomalies[0]["confidence"], 95)


class TestAbandonedAuthorizations(unittest.TestCase):

    def test_detects_abandoned_on_completed_ride(self):
        """Completed ride with auth but no capture = money lost."""
        rides = make_ride(status="completed")
        txns = make_txns([
            {"transaction_id": "T1", "ride_id": "RIDE-TEST", "event_type": "authorization",
             "amount": 200.0, "currency": "MXN", "timestamp": "2026-01-15T10:00:00",
             "status": "approved", "reference_txn_id": None},
        ])
        sessions = reconstruct_sessions(rides, txns, make_disputes())
        anomalies = detect_abandoned_authorizations(sessions)

        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0]["impact_category"], "money_lost")
        self.assertEqual(anomalies[0]["revenue_impact"], 195.0)  # actual_fare

    def test_cancelled_with_void_not_flagged(self):
        """Cancelled ride with proper void should NOT be flagged."""
        rides = make_ride(status="cancelled")
        txns = make_txns([
            {"transaction_id": "T1", "ride_id": "RIDE-TEST", "event_type": "authorization",
             "amount": 200.0, "currency": "MXN", "timestamp": "2026-01-15T10:00:00",
             "status": "approved", "reference_txn_id": None},
            {"transaction_id": "T2", "ride_id": "RIDE-TEST", "event_type": "void",
             "amount": 200.0, "currency": "MXN", "timestamp": "2026-01-15T10:10:00",
             "status": "approved", "reference_txn_id": "T1"},
        ])
        sessions = reconstruct_sessions(rides, txns, make_disputes())
        anomalies = detect_abandoned_authorizations(sessions)

        self.assertEqual(len(anomalies), 0)


class TestCurrencyDiscrepancies(unittest.TestCase):

    def test_detects_skewed_fx_rate(self):
        """Capture with USD amount deviating >2% from expected rate should be flagged."""
        rides = make_ride(actual=1720.0)  # ~$100 USD at MXN 17.2
        txns = make_txns([
            {"transaction_id": "T1", "ride_id": "RIDE-TEST", "event_type": "authorization",
             "amount": 1800.0, "currency": "MXN", "timestamp": "2026-01-15T10:00:00",
             "status": "approved", "reference_txn_id": None, "amount_usd": 104.65},
            {"transaction_id": "T2", "ride_id": "RIDE-TEST", "event_type": "capture",
             "amount": 1720.0, "currency": "MXN", "timestamp": "2026-01-15T10:30:00",
             "status": "approved", "reference_txn_id": "T1",
             "amount_usd": 120.0},  # Skewed: should be ~$100
        ])
        exchange_rates = pd.DataFrame([
            {"date": "2026-01-15", "currency": "MXN", "rate_to_usd": 17.2},
        ])
        sessions = reconstruct_sessions(rides, txns, make_disputes())
        anomalies = detect_currency_discrepancies(sessions, exchange_rates)

        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0]["anomaly_type"], "currency_discrepancy")

    def test_within_tolerance_not_flagged(self):
        """Small FX difference within 2% tolerance should NOT be flagged."""
        rides = make_ride(actual=1720.0)
        expected_usd = 1720.0 / 17.2  # 100.0
        txns = make_txns([
            {"transaction_id": "T1", "ride_id": "RIDE-TEST", "event_type": "authorization",
             "amount": 1800.0, "currency": "MXN", "timestamp": "2026-01-15T10:00:00",
             "status": "approved", "reference_txn_id": None, "amount_usd": 104.65},
            {"transaction_id": "T2", "ride_id": "RIDE-TEST", "event_type": "capture",
             "amount": 1720.0, "currency": "MXN", "timestamp": "2026-01-15T10:30:00",
             "status": "approved", "reference_txn_id": "T1",
             "amount_usd": expected_usd * 1.01},  # 1% off — within tolerance
        ])
        exchange_rates = pd.DataFrame([
            {"date": "2026-01-15", "currency": "MXN", "rate_to_usd": 17.2},
        ])
        sessions = reconstruct_sessions(rides, txns, make_disputes())
        anomalies = detect_currency_discrepancies(sessions, exchange_rates)

        self.assertEqual(len(anomalies), 0)


class TestDataValidation(unittest.TestCase):

    def test_quarantines_null_amounts(self):
        """Transactions with null amounts should be quarantined."""
        rides = make_ride()
        txns = make_txns([
            {"transaction_id": "T1", "ride_id": "RIDE-TEST", "event_type": "authorization",
             "amount": None, "currency": "MXN", "timestamp": "2026-01-15T10:00:00",
             "status": "approved", "reference_txn_id": None},
            {"transaction_id": "T2", "ride_id": "RIDE-TEST", "event_type": "capture",
             "amount": 195.0, "currency": "MXN", "timestamp": "2026-01-15T10:30:00",
             "status": "approved", "reference_txn_id": "T1"},
        ])
        disputes = make_disputes()
        _, cleaned_txns, _, report = validate_data(rides, txns, disputes)

        self.assertEqual(len(cleaned_txns), 1)  # null row removed
        self.assertGreater(report["quarantined_rows"], 0)

    def test_deduplicates_transaction_ids(self):
        """Duplicate transaction IDs should be deduplicated."""
        rides = make_ride()
        txns = make_txns([
            {"transaction_id": "T1", "ride_id": "RIDE-TEST", "event_type": "authorization",
             "amount": 200.0, "currency": "MXN", "timestamp": "2026-01-15T10:00:00",
             "status": "approved", "reference_txn_id": None},
            {"transaction_id": "T1", "ride_id": "RIDE-TEST", "event_type": "authorization",
             "amount": 200.0, "currency": "MXN", "timestamp": "2026-01-15T10:00:05",
             "status": "approved", "reference_txn_id": None},
        ])
        _, cleaned_txns, _, report = validate_data(rides, txns, make_disputes())

        self.assertEqual(len(cleaned_txns), 1)


if __name__ == "__main__":
    unittest.main()
