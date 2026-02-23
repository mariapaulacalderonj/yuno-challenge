"""
Apollo Rides - Transaction Anomaly Dashboard

Narrative Streamlit dashboard that tells the story of Apollo's revenue leak.
Flow: Headline -> Breakdown -> Location -> Trend -> Action

Run: streamlit run dashboard.py
"""

import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DATA_DIR = "data"

# -- Friendly labels for anomaly types --
ANOMALY_LABELS = {
    "duplicate_authorization": "Duplicate Authorization",
    "capture_mismatch": "Capture Mismatch",
    "ghost_refund": "Ghost Refund",
    "currency_discrepancy": "Currency Discrepancy",
    "abandoned_authorization": "Abandoned Authorization",
}

COUNTRY_LABELS = {"MX": "Mexico", "CO": "Colombia", "BR": "Brazil"}

COLOR_MAP = {
    "Duplicate Authorization": "#636EFA",
    "Capture Mismatch": "#EF553B",
    "Ghost Refund": "#00CC96",
    "Currency Discrepancy": "#AB63FA",
    "Abandoned Authorization": "#FFA15A",
}

IMPACT_COLORS = {"money_lost": "#EF553B", "money_at_risk": "#FFA15A"}


@st.cache_data
def load_data():
    """Load all pipeline outputs."""
    try:
        anomalies = pd.read_csv(f"{DATA_DIR}/anomalies.csv")
        rides = pd.read_csv(f"{DATA_DIR}/rides.csv")
        transactions = pd.read_csv(f"{DATA_DIR}/transactions.csv")
        validation = pd.read_csv(f"{DATA_DIR}/validation_report.csv")
    except FileNotFoundError:
        return None, None, None, None

    # Parse timestamps
    anomalies["detected_at"] = pd.to_datetime(anomalies["detected_at"])
    rides["timestamp"] = pd.to_datetime(rides["timestamp"])

    # Add friendly labels
    anomalies["anomaly_label"] = anomalies["anomaly_type"].map(ANOMALY_LABELS)
    anomalies["country_label"] = anomalies["country"].map(COUNTRY_LABELS)

    # Merge ride timestamp for trend analysis
    ride_ts = rides[["ride_id", "timestamp"]].rename(columns={"timestamp": "ride_timestamp"})
    anomalies = anomalies.merge(ride_ts, on="ride_id", how="left")
    anomalies["week"] = anomalies["ride_timestamp"].dt.to_period("W").dt.start_time

    return anomalies, rides, transactions, validation


def render_sidebar(anomalies):
    """Sidebar filters."""
    st.sidebar.header("Filters")

    # Anomaly type
    types = st.sidebar.multiselect(
        "Anomaly Type",
        options=sorted(anomalies["anomaly_label"].unique()),
        default=sorted(anomalies["anomaly_label"].unique()),
    )

    # Country
    countries = st.sidebar.multiselect(
        "Country",
        options=sorted(anomalies["country_label"].dropna().unique()),
        default=sorted(anomalies["country_label"].dropna().unique()),
    )

    # Confidence
    conf_min, conf_max = st.sidebar.slider(
        "Confidence Range",
        min_value=0, max_value=100,
        value=(0, 100),
    )

    # Impact category
    impact_cat = st.sidebar.multiselect(
        "Impact Category",
        options=["money_lost", "money_at_risk"],
        default=["money_lost", "money_at_risk"],
    )

    filtered = anomalies[
        (anomalies["anomaly_label"].isin(types))
        & (anomalies["country_label"].isin(countries))
        & (anomalies["confidence"] >= conf_min)
        & (anomalies["confidence"] <= conf_max)
        & (anomalies["impact_category"].isin(impact_cat))
    ]

    return filtered


def render_kpis(filtered, total_rides):
    """Section 1: Headline KPI metrics."""
    st.markdown("---")

    money_lost = filtered[filtered["impact_category"] == "money_lost"]["revenue_impact_usd"].sum()
    money_at_risk = filtered[filtered["impact_category"] == "money_at_risk"]["revenue_impact_usd"].sum()
    total_anomalies = len(filtered)
    affected_rides = filtered["ride_id"].nunique()
    pct_affected = (affected_rides / total_rides * 100) if total_rides > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Revenue Lost", f"${money_lost:,.2f}", help="Direct losses: ghost refunds, uncaptured fares, undercharges")
    col2.metric("Revenue at Risk", f"${money_at_risk:,.2f}", help="Potential losses: duplicate holds, overcharges, FX errors")
    col3.metric("Total Anomalies", f"{total_anomalies}")
    col4.metric("Affected Rides", f"{affected_rides} ({pct_affected:.1f}%)")


def render_breakdown(filtered):
    """Section 2: Where is the money going?"""
    st.markdown("---")
    st.subheader("Where is the money going?")

    col1, col2 = st.columns(2)

    with col1:
        type_counts = filtered["anomaly_label"].value_counts().reset_index()
        type_counts.columns = ["Anomaly Type", "Count"]
        fig = px.pie(
            type_counts, names="Anomaly Type", values="Count",
            color="Anomaly Type", color_discrete_map=COLOR_MAP,
            title="Anomalies by Type",
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(showlegend=False, margin=dict(t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        impact_by_type = (
            filtered.groupby(["anomaly_label", "impact_category"])["revenue_impact_usd"]
            .sum()
            .reset_index()
        )
        impact_by_type.columns = ["Anomaly Type", "Impact Category", "USD"]
        fig = px.bar(
            impact_by_type, x="Anomaly Type", y="USD",
            color="Impact Category",
            color_discrete_map=IMPACT_COLORS,
            title="Revenue Impact by Type (USD)",
            barmode="stack",
        )
        fig.update_layout(margin=dict(t=40, b=0), xaxis_tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)


def render_location(filtered):
    """Section 3: Where is the problem?"""
    st.markdown("---")
    st.subheader("Where is the problem?")

    col1, col2 = st.columns(2)

    with col1:
        country_type = (
            filtered.groupby(["country_label", "anomaly_label"])
            .size()
            .reset_index(name="Count")
        )
        fig = px.bar(
            country_type, x="country_label", y="Count",
            color="anomaly_label",
            color_discrete_map=COLOR_MAP,
            title="Anomalies by Country",
            barmode="stack",
            labels={"country_label": "Country", "anomaly_label": "Type"},
        )
        fig.update_layout(margin=dict(t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        country_impact = (
            filtered.groupby(["country_label", "impact_category"])["revenue_impact_usd"]
            .sum()
            .reset_index()
        )
        country_impact.columns = ["Country", "Impact Category", "USD"]
        fig = px.bar(
            country_impact, x="Country", y="USD",
            color="Impact Category",
            color_discrete_map=IMPACT_COLORS,
            title="Revenue Impact by Country (USD)",
            barmode="stack",
        )
        fig.update_layout(margin=dict(t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)


def render_trend(filtered):
    """Section 4: Is it getting worse?"""
    st.markdown("---")
    st.subheader("Is it getting worse?")

    weekly = (
        filtered.groupby(["week", "anomaly_label"])
        .agg(count=("ride_id", "size"), usd=("revenue_impact_usd", "sum"))
        .reset_index()
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(
            weekly, x="week", y="count",
            color="anomaly_label",
            color_discrete_map=COLOR_MAP,
            title="Weekly Anomaly Count",
            labels={"week": "Week", "count": "Anomalies", "anomaly_label": "Type"},
            markers=True,
        )
        fig.update_layout(margin=dict(t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.line(
            weekly, x="week", y="usd",
            color="anomaly_label",
            color_discrete_map=COLOR_MAP,
            title="Weekly Revenue Impact (USD)",
            labels={"week": "Week", "usd": "USD Impact", "anomaly_label": "Type"},
            markers=True,
        )
        fig.update_layout(margin=dict(t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)


def render_confidence(filtered):
    """Section 5: Confidence distribution."""
    st.markdown("---")
    st.subheader("Confidence Distribution")

    fig = px.histogram(
        filtered, x="confidence", color="anomaly_label",
        color_discrete_map=COLOR_MAP,
        nbins=20, barmode="overlay", opacity=0.7,
        title="How certain are we about each anomaly?",
        labels={"confidence": "Confidence Score (%)", "anomaly_label": "Type"},
    )
    fig.update_layout(margin=dict(t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)


def render_action_table(filtered):
    """Section 6: What to investigate — drill-down table."""
    st.markdown("---")
    st.subheader("Transactions to Investigate")

    display_df = filtered[[
        "ride_id", "anomaly_label", "impact_category", "country_label",
        "currency", "revenue_impact", "revenue_impact_usd", "confidence",
        "recommendation",
    ]].copy()
    display_df.columns = [
        "Ride ID", "Anomaly Type", "Impact", "Country",
        "Currency", "Amount (Local)", "Amount (USD)", "Confidence",
        "Recommendation",
    ]
    display_df = display_df.sort_values("Confidence", ascending=False)

    st.dataframe(
        display_df.drop(columns=["Recommendation"]),
        use_container_width=True,
        height=400,
    )

    # Top anomalies with expandable details
    st.markdown("#### Top Anomalies — Details & Recommendations")
    top_n = min(15, len(filtered))
    for _, row in filtered.nlargest(top_n, "confidence").iterrows():
        with st.expander(
            f"{row['anomaly_label']} — {row['ride_id']} "
            f"({row['confidence']}% confidence, ${row['revenue_impact_usd']:.2f} USD)"
        ):
            st.markdown(f"**Impact:** {row['impact_category'].replace('_', ' ').title()}")
            st.markdown(f"**Recommendation:** {row['recommendation']}")
            try:
                details = json.loads(row["details"])
                st.json(details)
            except (json.JSONDecodeError, TypeError):
                st.text(str(row["details"]))


def render_insights(filtered):
    """Section 7: Key insights and root cause hypotheses."""
    st.markdown("---")
    st.subheader("Key Insights & Root Cause Hypotheses")

    if filtered.empty:
        st.info("No anomalies match current filters.")
        return

    # Find the biggest problem
    type_impact = filtered.groupby("anomaly_label")["revenue_impact_usd"].sum()
    worst_type = type_impact.idxmax()
    worst_amount = type_impact.max()

    country_impact = filtered.groupby("country_label")["revenue_impact_usd"].sum()
    worst_country = country_impact.idxmax()
    worst_country_amount = country_impact.max()

    money_lost = filtered[filtered["impact_category"] == "money_lost"]["revenue_impact_usd"].sum()
    total_impact = filtered["revenue_impact_usd"].sum()
    lost_pct = (money_lost / total_impact * 100) if total_impact > 0 else 0

    insights = [
        f"**{worst_type}** accounts for the largest revenue impact at **${worst_amount:,.2f} USD** "
        f"({worst_amount/total_impact*100:.0f}% of total).",
        f"**{worst_country}** is the most affected market with **${worst_country_amount:,.2f} USD** at risk.",
        f"**{lost_pct:.0f}%** of the total impact (${money_lost:,.2f}) is confirmed **money lost** "
        f"(ghost refunds + abandoned auths + undercharges).",
    ]

    # Currency-specific insight with root cause
    if "Currency Discrepancy" in filtered["anomaly_label"].values:
        fx_df = filtered[filtered["anomaly_label"] == "Currency Discrepancy"]
        fx_by_currency = fx_df.groupby("currency").size()
        if len(fx_by_currency) > 0:
            worst_fx_currency = fx_by_currency.idxmax()
            fx_count = fx_by_currency.max()
            total_fx = len(fx_df)
            insights.append(
                f"Currency discrepancies are **{fx_count/total_fx*100:.0f}%** concentrated in "
                f"**{worst_fx_currency}** transactions ({fx_count}/{total_fx} cases) "
                f"→ likely a processor configuration issue with FX rate source for {worst_fx_currency}. "
                f"**Action:** audit the payment processor's exchange rate feed and compare against market rates."
            )

    # Ghost refund insight with root cause
    if "Ghost Refund" in filtered["anomaly_label"].values:
        ghost_df = filtered[filtered["anomaly_label"] == "Ghost Refund"]
        ghost_by_country = ghost_df.groupby("country_label")["revenue_impact_usd"].sum()
        if len(ghost_by_country) > 0:
            worst_ghost_country = ghost_by_country.idxmax()
            ghost_pct = ghost_by_country.max() / ghost_by_country.sum() * 100
            insights.append(
                f"**{ghost_pct:.0f}%** of ghost refund losses are in **{worst_ghost_country}** "
                f"(${ghost_by_country.max():,.2f} USD) "
                f"→ possible compromised refund workflow or misconfigured automated refund rules in this market. "
                f"**Action:** audit refund authorization permissions and check for unauthorized API access."
            )

    # Duplicate auth insight with root cause
    if "Duplicate Authorization" in filtered["anomaly_label"].values:
        dup_df = filtered[filtered["anomaly_label"] == "Duplicate Authorization"]
        if len(dup_df) > 0:
            dup_impact = dup_df["revenue_impact_usd"].sum()
            high_conf_dup = len(dup_df[dup_df["confidence"] >= 90])
            insights.append(
                f"**{high_conf_dup}/{len(dup_df)}** duplicate authorizations are high-confidence (≥90%) "
                f"with **${dup_impact:,.2f} USD** in held funds "
                f"→ strongly indicates client-side payment retry logic sending duplicate requests. "
                f"**Action:** implement idempotency keys on the payment gateway and add dedup logic at the API layer."
            )

    # Abandoned auth insight with root cause
    if "Abandoned Authorization" in filtered["anomaly_label"].values:
        aband_df = filtered[filtered["anomaly_label"] == "Abandoned Authorization"]
        if len(aband_df) > 0:
            aband_lost = aband_df[aband_df["impact_category"] == "money_lost"]
            if len(aband_lost) > 0:
                insights.append(
                    f"**{len(aband_lost)} completed rides** were never captured — **${aband_lost['revenue_impact_usd'].sum():,.2f} USD** "
                    f"of earned revenue was never collected "
                    f"→ the ride-completion-to-capture webhook is failing silently. "
                    f"**Action:** add monitoring on the capture callback and implement a reconciliation cron job "
                    f"that catches uncaptured rides within 24h."
                )

    for i, insight in enumerate(insights, 1):
        st.markdown(f"{i}. {insight}")

    # Recommended next steps summary
    st.markdown("---")
    st.subheader("Recommended Next Steps")
    st.markdown("""
1. **Immediate recovery** — Process the high-confidence ghost refund and abandoned authorization cases to recover confirmed lost revenue
2. **Duplicate auth prevention** — Deploy idempotency keys on the payment gateway to prevent duplicate authorizations at the source
3. **Capture monitoring** — Set up alerts for rides that remain uncaptured >1 hour after completion
4. **FX rate audit** — Compare processor-applied exchange rates against market benchmarks for each currency
5. **Refund controls** — Implement approval workflows for refunds on completed rides with no dispute record
    """)


def render_data_quality(validation):
    """Section 8: Data quality report."""
    st.markdown("---")
    st.subheader("Data Quality Report")

    if validation is not None and len(validation) > 0:
        issues = validation["issue"].tolist()
        if issues == ["No issues"]:
            st.success("No data quality issues detected in the pipeline input.")
        else:
            st.warning(f"{len(issues)} data quality issue(s) found:")
            for issue in issues:
                st.markdown(f"- {issue}")
    else:
        st.info("No validation report available.")


def main():
    st.set_page_config(
        page_title="Apollo Rides - Anomaly Detector",
        page_icon="🔍",
        layout="wide",
    )

    st.title("Apollo Rides — Transaction Anomaly Detector")
    st.caption("Identifying revenue leaks across Mexico, Colombia, and Brazil")

    anomalies, rides, transactions, validation = load_data()

    if anomalies is None or anomalies.empty:
        st.error(
            "No data found. Run the pipeline first:\n\n"
            "```bash\n"
            "python generate_data.py\n"
            "python detect_anomalies.py\n"
            "```"
        )
        return

    total_rides = len(rides) if rides is not None else 0
    filtered = render_sidebar(anomalies)

    if filtered.empty:
        st.warning("No anomalies match the current filters. Adjust the sidebar filters.")
        return

    render_kpis(filtered, total_rides)
    render_breakdown(filtered)
    render_location(filtered)
    render_trend(filtered)
    render_confidence(filtered)
    render_action_table(filtered)
    render_insights(filtered)
    render_data_quality(validation)


if __name__ == "__main__":
    main()
