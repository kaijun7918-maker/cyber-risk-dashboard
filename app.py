"""Interactive Streamlit dashboard for the cyber-risk project."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


st.set_page_config(
    page_title="Global Cyber Risk Analytics",
    page_icon="🛡️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


@st.cache_data
def load_project_data() -> tuple[pd.DataFrame, ...]:
    prepared = pd.read_csv(DATA_DIR / "cyber_incidents_prepared.csv")
    segmented = pd.read_csv(DATA_DIR / "cyber_incidents_segmented.csv")
    predictions = pd.read_csv(DATA_DIR / "cyber_model_predictions.csv")
    profiles = pd.read_csv(DATA_DIR / "cyber_risk_profile_summary.csv")
    return prepared, segmented, predictions, profiles


def default_api_url() -> str:
    try:
        if "API_URL" in st.secrets:
            return str(st.secrets["API_URL"])
    except Exception:
        # Local development can run without a secrets.toml file.
        pass
    return os.getenv("API_URL", "http://localhost:8000")


def calculate_metrics(predictions: pd.DataFrame) -> tuple[float, float, float]:
    actual = predictions["resolution_time_hours"].astype(float)
    predicted = predictions["predicted_resolution_time"].astype(float)
    errors = actual - predicted
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    denominator = float(np.sum(np.square(actual - actual.mean())))
    r2 = 1 - float(np.sum(np.square(errors))) / denominator
    return mae, rmse, r2


prepared_df, segmented_df, predictions_df, profiles_df = load_project_data()

st.title("Global Cybersecurity Risk Analytics")
st.caption(
    "Interactive incident trends, Spark ML Risk Profiles and "
    "Random Forest resolution-time predictions"
)

with st.sidebar:
    st.header("Dashboard Filters")

    minimum_year = int(prepared_df["year"].min())
    maximum_year = int(prepared_df["year"].max())
    selected_years = st.slider(
        "Incident year",
        minimum_year,
        maximum_year,
        (minimum_year, maximum_year),
    )

    all_countries = sorted(prepared_df["country"].dropna().unique())
    selected_countries = st.multiselect(
        "Country",
        all_countries,
        default=all_countries,
    )

    all_attack_types = sorted(prepared_df["attack_type"].dropna().unique())
    selected_attack_types = st.multiselect(
        "Attack type",
        all_attack_types,
        default=all_attack_types,
    )

    st.divider()
    api_url = st.text_input(
        "FastAPI base URL",
        value=default_api_url(),
        help="Local: http://localhost:8000 or your Render URL",
    ).strip().rstrip("/")


filter_mask = (
    prepared_df["year"].between(*selected_years)
    & prepared_df["country"].isin(selected_countries)
    & prepared_df["attack_type"].isin(selected_attack_types)
)
filtered_df = prepared_df.loc[filter_mask].copy()

segmented_mask = (
    segmented_df["year"].between(*selected_years)
    & segmented_df["country"].isin(selected_countries)
    & segmented_df["attack_type"].isin(selected_attack_types)
)
filtered_segmented_df = segmented_df.loc[segmented_mask].copy()

if filtered_df.empty:
    st.warning("No incidents match the selected filters.")
    st.stop()

overview_tab, trends_tab, profiles_tab, evaluation_tab, prediction_tab = st.tabs(
    [
        "Overview",
        "Trends and Patterns",
        "Risk Profiles",
        "Model Evaluation",
        "Predict Resolution Time",
    ]
)

with overview_tab:
    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Incidents", f"{len(filtered_df):,}")
    metric_2.metric(
        "Average Financial Loss",
        f"${filtered_df['financial_loss_million'].mean():,.2f}M",
    )
    metric_3.metric(
        "Average Affected Users",
        f"{filtered_df['affected_users'].mean():,.0f}",
    )
    metric_4.metric(
        "Average Resolution Time",
        f"{filtered_df['resolution_time_hours'].mean():.2f} hours",
    )

    country_summary = (
        filtered_df.groupby("country", as_index=False)
        .agg(
            incidents=("attack_type", "size"),
            total_financial_loss_million=("financial_loss_million", "sum"),
        )
        .sort_values("total_financial_loss_million", ascending=False)
    )

    overview_chart = px.bar(
        country_summary,
        x="country",
        y="total_financial_loss_million",
        color="incidents",
        labels={
            "country": "Country",
            "total_financial_loss_million": "Total Financial Loss (Million USD)",
            "incidents": "Incidents",
        },
        title="Financial Impact by Country",
    )
    st.plotly_chart(overview_chart, use_container_width=True)

with trends_tab:
    yearly_summary = (
        filtered_df.groupby("year", as_index=False)
        .agg(
            total_financial_loss_million=("financial_loss_million", "sum"),
            average_resolution_time_hours=("resolution_time_hours", "mean"),
            incidents=("attack_type", "size"),
        )
        .sort_values("year")
    )

    financial_trend = px.line(
        yearly_summary,
        x="year",
        y="total_financial_loss_million",
        markers=True,
        labels={
            "year": "Year",
            "total_financial_loss_million": "Total Financial Loss (Million USD)",
        },
        title="Financial Loss over Time",
    )
    st.plotly_chart(financial_trend, use_container_width=True)

    trend_col_1, trend_col_2 = st.columns(2)

    attack_distribution = (
        filtered_df["attack_type"]
        .value_counts()
        .rename_axis("attack_type")
        .reset_index(name="number_of_incidents")
    )
    attack_chart = px.bar(
        attack_distribution,
        x="attack_type",
        y="number_of_incidents",
        color="number_of_incidents",
        labels={
            "attack_type": "Attack Type",
            "number_of_incidents": "Number of Incidents",
        },
        title="Attack-Type Distribution",
    )
    trend_col_1.plotly_chart(attack_chart, use_container_width=True)

    industry_loss = (
        filtered_df.groupby("target_industry", as_index=False)
        ["financial_loss_million"]
        .sum()
        .sort_values("financial_loss_million", ascending=False)
    )
    industry_chart = px.bar(
        industry_loss,
        x="target_industry",
        y="financial_loss_million",
        color="financial_loss_million",
        labels={
            "target_industry": "Target Industry",
            "financial_loss_million": "Total Financial Loss (Million USD)",
        },
        title="Financial Loss by Target Industry",
    )
    trend_col_2.plotly_chart(industry_chart, use_container_width=True)

with profiles_tab:
    profile_counts = (
        filtered_segmented_df["risk_profile"]
        .value_counts()
        .rename_axis("risk_profile")
        .reset_index(name="number_of_incidents")
    )

    profile_col_1, profile_col_2 = st.columns([1, 1.5])

    profile_pie = px.pie(
        profile_counts,
        names="risk_profile",
        values="number_of_incidents",
        hole=0.42,
        title="Risk Profile Breakdown",
    )
    profile_col_1.plotly_chart(profile_pie, use_container_width=True)

    profile_scatter = px.scatter(
        filtered_segmented_df,
        x="affected_users",
        y="financial_loss_million",
        color="risk_profile",
        hover_data=["country", "attack_type", "target_industry"],
        opacity=0.7,
        labels={
            "affected_users": "Affected Users",
            "financial_loss_million": "Financial Loss (Million USD)",
            "risk_profile": "Risk Profile",
        },
        title="Incident Segmentation by Financial and User Impact",
    )
    profile_col_2.plotly_chart(profile_scatter, use_container_width=True)

    st.subheader("Risk Profile Summary")
    st.dataframe(
        profiles_df,
        use_container_width=True,
        hide_index=True,
    )

with evaluation_tab:
    mae, rmse, r2 = calculate_metrics(predictions_df)
    evaluation_1, evaluation_2, evaluation_3 = st.columns(3)
    evaluation_1.metric("Testing MAE", f"{mae:.2f} hours")
    evaluation_2.metric("Testing RMSE", f"{rmse:.2f} hours")
    evaluation_3.metric("Testing R²", f"{r2:.4f}")

    evaluation_col_1, evaluation_col_2 = st.columns(2)

    actual_prediction_chart = px.scatter(
        predictions_df,
        x="resolution_time_hours",
        y="predicted_resolution_time",
        color="attack_type",
        opacity=0.7,
        labels={
            "resolution_time_hours": "Actual Resolution Time (Hours)",
            "predicted_resolution_time": "Predicted Resolution Time (Hours)",
        },
        title="Actual versus Predicted Resolution Time",
    )
    lower_bound = float(
        min(
            predictions_df["resolution_time_hours"].min(),
            predictions_df["predicted_resolution_time"].min(),
        )
    )
    upper_bound = float(
        max(
            predictions_df["resolution_time_hours"].max(),
            predictions_df["predicted_resolution_time"].max(),
        )
    )
    actual_prediction_chart.add_trace(
        go.Scatter(
            x=[lower_bound, upper_bound],
            y=[lower_bound, upper_bound],
            mode="lines",
            name="Perfect Prediction",
            line={"color": "red", "dash": "dash"},
        )
    )
    evaluation_col_1.plotly_chart(
        actual_prediction_chart,
        use_container_width=True,
    )

    residual_chart = px.histogram(
        predictions_df,
        x="residual",
        nbins=20,
        color_discrete_sequence=["#e67e22"],
        labels={"residual": "Residual: Actual - Predicted (Hours)"},
        title="Residual Distribution",
    )
    residual_chart.add_vline(
        x=0,
        line_dash="dash",
        line_color="red",
    )
    evaluation_col_2.plotly_chart(
        residual_chart,
        use_container_width=True,
    )

    st.warning(
        "The negative testing R² indicates limited predictive signal in the "
        "available incident characteristics. Predictions should be treated as "
        "decision-support estimates rather than precise operational forecasts."
    )

with prediction_tab:
    st.subheader("Predict Incident Resolution Time")
    st.write(
        "Enter the characteristics of a new incident. The application sends "
        "the input to the deployed FastAPI service on Render."
    )

    if st.button("Check API Health"):
        try:
            health_response = requests.get(
                f"{api_url}/health",
                timeout=30,
            )
            health_response.raise_for_status()
            st.success("FastAPI service is healthy.")
            st.json(health_response.json())
        except requests.RequestException as exc:
            st.error(f"API health check failed: {exc}")

    with st.form("prediction_form"):
        form_col_1, form_col_2, form_col_3 = st.columns(3)

        with form_col_1:
            country = st.selectbox(
                "Country",
                sorted(prepared_df["country"].unique()),
            )
            year = st.number_input(
                "Year",
                min_value=2015,
                max_value=2024,
                value=2024,
                step=1,
            )
            attack_type = st.selectbox(
                "Attack Type",
                sorted(prepared_df["attack_type"].unique()),
            )

        with form_col_2:
            target_industry = st.selectbox(
                "Target Industry",
                sorted(prepared_df["target_industry"].unique()),
            )
            financial_loss = st.number_input(
                "Financial Loss (Million USD)",
                min_value=0.0,
                value=float(prepared_df["financial_loss_million"].median()),
                step=1.0,
            )
            affected_users = st.number_input(
                "Affected Users",
                min_value=1,
                value=int(prepared_df["affected_users"].median()),
                step=1000,
            )

        with form_col_3:
            attack_source = st.selectbox(
                "Attack Source",
                sorted(prepared_df["attack_source"].unique()),
            )
            vulnerability_type = st.selectbox(
                "Vulnerability Type",
                sorted(prepared_df["vulnerability_type"].unique()),
            )
            defense_mechanism = st.selectbox(
                "Defense Mechanism",
                sorted(prepared_df["defense_mechanism"].unique()),
            )

        submitted = st.form_submit_button(
            "Predict Resolution Time",
            use_container_width=True,
        )

    if submitted:
        payload = {
            "country": country,
            "year": int(year),
            "attack_type": attack_type,
            "target_industry": target_industry,
            "financial_loss_million": float(financial_loss),
            "affected_users": int(affected_users),
            "attack_source": attack_source,
            "vulnerability_type": vulnerability_type,
            "defense_mechanism": defense_mechanism,
        }

        with st.expander("Prediction request"):
            st.json(payload)

        try:
            with st.spinner("Requesting prediction from FastAPI..."):
                response = requests.post(
                    f"{api_url}/predict",
                    json=payload,
                    timeout=180,
                )
                response.raise_for_status()
                result = response.json()

            st.success("Prediction completed successfully.")
            result_col_1, result_col_2 = st.columns(2)
            result_col_1.metric(
                "Predicted Resolution Time",
                f"{result['predicted_resolution_time_hours']:.2f} hours",
            )
            result_col_2.metric(
                "Equivalent Duration",
                f"{result['predicted_resolution_time_days']:.2f} days",
            )
            st.caption(
                f"Model: {result['model_type']} | "
                f"Spark: {result['spark_version']}"
            )
            st.info(
                "This estimate should be interpreted together with the "
                "Part I evaluation results and residual analysis."
            )

        except requests.Timeout:
            st.error(
                "The API request timed out. A sleeping Render service may "
                "need additional time to start its Spark session."
            )
        except requests.RequestException as exc:
            error_message = str(exc)
            if getattr(exc, "response", None) is not None:
                error_message = exc.response.text
            st.error(f"Prediction request failed: {error_message}")