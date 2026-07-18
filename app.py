from __future__ import annotations

import math
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


st.set_page_config(
    page_title="Cyber Incident Risk Dashboard",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_API_URL = "https://cyber-incident-resolution-api.onrender.com"

try:
    API_URL = st.secrets.get(
        "API_URL",
        os.getenv("API_URL", DEFAULT_API_URL),
    )
except Exception:
    API_URL = os.getenv("API_URL", DEFAULT_API_URL)

API_URL = str(API_URL).rstrip("/")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {
        background: rgba(37, 99, 235, 0.06);
        border: 1px solid rgba(37, 99, 235, 0.16);
        border-radius: 12px;
        padding: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data(data_directory: str):
    data_path = Path(data_directory)
    paths = {
        "prepared": data_path / "cyber_incidents_prepared.csv",
        "segmented": data_path / "cyber_incidents_segmented.csv",
        "summary": data_path / "cyber_risk_profile_summary.csv",
        "predictions": data_path / "cyber_model_predictions.csv",
    }

    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing files: " + ", ".join(missing))

    return (
        pd.read_csv(paths["prepared"]),
        pd.read_csv(paths["segmented"]),
        pd.read_csv(paths["summary"]),
        pd.read_csv(paths["predictions"]),
    )


def require_columns(df, columns, file_name):
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"{file_name} is missing: {', '.join(sorted(missing))}"
        )


def apply_filters(df, year_range, countries, attack_types):
    result = df[df["year"].between(year_range[0], year_range[1])].copy()
    if countries:
        result = result[result["country"].isin(countries)]
    if attack_types:
        result = result[result["attack_type"].isin(attack_types)]
    return result


def finish_figure(fig):
    fig.update_layout(
        margin=dict(l=20, r=20, t=55, b=20),
        legend_title_text="",
        hoverlabel=dict(bgcolor="white"),
    )
    return fig


try:
    prepared_df, segmented_df, risk_summary_df, predictions_df = load_data(
        str(DATA_DIR)
    )

    require_columns(
        prepared_df,
        [
            "country",
            "year",
            "attack_type",
            "target_industry",
            "financial_loss_million",
            "affected_users",
            "attack_source",
            "vulnerability_type",
            "defense_mechanism",
            "resolution_time_hours",
            "impact_level",
        ],
        "cyber_incidents_prepared.csv",
    )
    require_columns(
        segmented_df,
        [
            "country",
            "year",
            "attack_type",
            "financial_loss_million",
            "affected_users",
            "resolution_time_hours",
            "cluster",
            "risk_profile",
        ],
        "cyber_incidents_segmented.csv",
    )
    require_columns(
        predictions_df,
        [
            "country",
            "year",
            "attack_type",
            "resolution_time_hours",
            "predicted_resolution_time",
        ],
        "cyber_model_predictions.csv",
    )
except Exception as error:
    st.error(f"Dashboard data could not be loaded: {error}")
    st.info("Place the four CSV files inside cyber-risk-dashboard/data/.")
    st.stop()

for dataframe in (prepared_df, segmented_df, predictions_df):
    dataframe["year"] = pd.to_numeric(dataframe["year"], errors="coerce")


st.sidebar.title("Dashboard Controls")
st.sidebar.subheader("Global Filters")

minimum_year = int(prepared_df["year"].min())
maximum_year = int(prepared_df["year"].max())

year_range = st.sidebar.slider(
    "Incident year",
    minimum_year,
    maximum_year,
    (minimum_year, maximum_year),
)

country_options = sorted(prepared_df["country"].dropna().unique().tolist())
attack_options = sorted(prepared_df["attack_type"].dropna().unique().tolist())

selected_countries = st.sidebar.multiselect(
    "Countries",
    country_options,
    placeholder="All countries",
)
selected_attacks = st.sidebar.multiselect(
    "Attack types",
    attack_options,
    placeholder="All attack types",
)
st.sidebar.caption("An empty selection includes every category.")

prepared_filtered = apply_filters(
    prepared_df,
    year_range,
    selected_countries,
    selected_attacks,
)
segmented_filtered = apply_filters(
    segmented_df,
    year_range,
    selected_countries,
    selected_attacks,
)
predictions_filtered = apply_filters(
    predictions_df,
    year_range,
    selected_countries,
    selected_attacks,
)

st.sidebar.divider()
st.sidebar.subheader("FastAPI Service")
st.sidebar.caption(API_URL)

if st.sidebar.button("Check API Connection"):
    try:
        with st.spinner("Waking and checking the Render API..."):
            health_response = requests.get(f"{API_URL}/health", timeout=120)
            health_response.raise_for_status()
            health_data = health_response.json()
        st.sidebar.success(
            f"API online · Model loaded: {health_data.get('model_loaded', False)}"
        )
    except requests.RequestException as error:
        st.sidebar.error(f"API connection failed: {error}")

st.sidebar.markdown(f"[Open FastAPI documentation]({API_URL}/docs)")


st.title("Cyber Incident Risk Analytics Dashboard")
st.caption(
    "Global cybersecurity incidents from 2015–2024, Spark MLlib risk "
    "profiles, Random Forest evaluation, and live resolution-time prediction."
)

if prepared_filtered.empty:
    st.warning("No incidents match the current filters.")
    st.stop()

overview_tab, risk_tab, model_tab, prediction_tab = st.tabs(
    [
        "Overview and Trends",
        "Risk Profiles",
        "Model Performance",
        "Resolution-Time Prediction",
    ]
)


with overview_tab:
    st.subheader("Incident Overview")

    metric_columns = st.columns(4)
    metric_columns[0].metric("Incidents", f"{len(prepared_filtered):,}")
    metric_columns[1].metric(
        "Total Financial Loss",
        f"${prepared_filtered['financial_loss_million'].sum():,.2f}M",
    )
    metric_columns[2].metric(
        "Average Affected Users",
        f"{prepared_filtered['affected_users'].mean():,.0f}",
    )
    metric_columns[3].metric(
        "Average Resolution Time",
        f"{prepared_filtered['resolution_time_hours'].mean():.2f} hours",
    )

    yearly_df = (
        prepared_filtered.groupby("year", as_index=False)
        .agg(
            total_financial_loss_million=("financial_loss_million", "sum"),
            average_resolution_time_hours=("resolution_time_hours", "mean"),
            number_of_incidents=("attack_type", "size"),
        )
        .sort_values("year")
    )

    attack_distribution_df = (
        prepared_filtered["attack_type"]
        .value_counts()
        .rename_axis("attack_type")
        .reset_index(name="number_of_incidents")
    )

    chart_left, chart_right = st.columns(2)

    with chart_left:
        loss_figure = px.line(
            yearly_df,
            x="year",
            y="total_financial_loss_million",
            markers=True,
            title="Total Financial Loss Over Time",
            labels={
                "year": "Year",
                "total_financial_loss_million": "Financial Loss (Million USD)",
            },
        )
        loss_figure.update_traces(line_color="#2563EB", line_width=3)
        st.plotly_chart(finish_figure(loss_figure), use_container_width=True)

    with chart_right:
        attack_figure = px.bar(
            attack_distribution_df,
            x="number_of_incidents",
            y="attack_type",
            orientation="h",
            color="number_of_incidents",
            color_continuous_scale="Blues",
            title="Attack-Type Distribution",
            labels={
                "number_of_incidents": "Number of Incidents",
                "attack_type": "Attack Type",
            },
        )
        attack_figure.update_layout(
            coloraxis_showscale=False,
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(finish_figure(attack_figure), use_container_width=True)

    industry_df = (
        prepared_filtered.groupby("target_industry", as_index=False)
        .agg(
            total_financial_loss_million=("financial_loss_million", "sum"),
            average_resolution_time_hours=("resolution_time_hours", "mean"),
            number_of_incidents=("attack_type", "size"),
        )
        .sort_values("total_financial_loss_million", ascending=False)
    )

    impact_df = (
        prepared_filtered["impact_level"]
        .value_counts()
        .rename_axis("impact_level")
        .reset_index(name="number_of_incidents")
    )

    pattern_left, pattern_right = st.columns(2)

    with pattern_left:
        industry_figure = px.bar(
            industry_df,
            x="target_industry",
            y="total_financial_loss_million",
            color="average_resolution_time_hours",
            color_continuous_scale="OrRd",
            title="Financial Loss by Target Industry",
            labels={
                "target_industry": "Target Industry",
                "total_financial_loss_million": "Total Loss (Million USD)",
                "average_resolution_time_hours": "Average Resolution Time",
            },
        )
        industry_figure.update_xaxes(tickangle=-25)
        st.plotly_chart(finish_figure(industry_figure), use_container_width=True)

    with pattern_right:
        impact_figure = px.pie(
            impact_df,
            names="impact_level",
            values="number_of_incidents",
            hole=0.48,
            title="Impact-Level Distribution",
            color="impact_level",
            color_discrete_map={
                "High": "#DC2626",
                "Medium": "#F59E0B",
                "Low": "#16A34A",
            },
        )
        impact_figure.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(finish_figure(impact_figure), use_container_width=True)

    with st.expander("View filtered incident records"):
        st.dataframe(prepared_filtered, use_container_width=True, hide_index=True)


with risk_tab:
    st.subheader("Spark MLlib Incident Segmentation")
    st.write(
        "K-Means selected four clusters using standardized financial loss and "
        "affected-user features. The descriptive labels reflect each cluster's "
        "economic impact, user impact, and most common attack type."
    )

    if segmented_filtered.empty:
        st.warning("No segmented incidents match the current filters.")
    else:
        risk_aggregate_df = (
            segmented_filtered.groupby(
                ["cluster", "risk_profile"],
                as_index=False,
            )
            .agg(
                number_of_incidents=("attack_type", "size"),
                average_financial_loss_million=("financial_loss_million", "mean"),
                average_affected_users=("affected_users", "mean"),
                average_resolution_time_hours=("resolution_time_hours", "mean"),
            )
            .sort_values("cluster")
        )

        top_attack_df = (
            segmented_filtered.groupby(
                ["cluster", "risk_profile", "attack_type"],
                as_index=False,
            )
            .size()
            .rename(columns={"size": "top_attack_count"})
            .sort_values(
                ["cluster", "top_attack_count", "attack_type"],
                ascending=[True, False, True],
            )
            .drop_duplicates(["cluster", "risk_profile"])
            .rename(columns={"attack_type": "common_attack_type"})
        )

        risk_aggregate_df = risk_aggregate_df.merge(
            top_attack_df,
            on=["cluster", "risk_profile"],
            how="left",
        )

        risk_left, risk_right = st.columns(2)

        with risk_left:
            risk_pie = px.pie(
                risk_aggregate_df,
                names="risk_profile",
                values="number_of_incidents",
                hole=0.42,
                title="Risk-Profile Breakdown",
            )
            risk_pie.update_traces(
                textinfo="percent",
                hovertemplate=(
                    "%{label}<br>Incidents: %{value}<br>Share: %{percent}"
                    "<extra></extra>"
                ),
            )
            st.plotly_chart(finish_figure(risk_pie), use_container_width=True)

        with risk_right:
            risk_scatter = px.scatter(
                risk_aggregate_df,
                x="average_financial_loss_million",
                y="average_affected_users",
                size="number_of_incidents",
                color="risk_profile",
                hover_name="risk_profile",
                hover_data={
                    "cluster": True,
                    "number_of_incidents": True,
                    "common_attack_type": True,
                    "average_financial_loss_million": ":.2f",
                    "average_affected_users": ":,.0f",
                },
                title="Risk Profiles by Financial and User Impact",
                labels={
                    "average_financial_loss_million": (
                        "Average Financial Loss (Million USD)"
                    ),
                    "average_affected_users": "Average Affected Users",
                },
            )
            risk_scatter.update_traces(
                marker=dict(sizemin=18, line=dict(width=1, color="white"))
            )
            st.plotly_chart(finish_figure(risk_scatter), use_container_width=True)

        risk_table = risk_aggregate_df[
            [
                "cluster",
                "risk_profile",
                "number_of_incidents",
                "average_financial_loss_million",
                "average_affected_users",
                "average_resolution_time_hours",
                "common_attack_type",
                "top_attack_count",
            ]
        ].copy()
        numeric_columns = [
            "average_financial_loss_million",
            "average_affected_users",
            "average_resolution_time_hours",
        ]
        risk_table[numeric_columns] = risk_table[numeric_columns].round(2)
        st.dataframe(risk_table, use_container_width=True, hide_index=True)

        with st.expander("View original Part I risk-profile summary"):
            st.dataframe(risk_summary_df, use_container_width=True, hide_index=True)


with model_tab:
    st.subheader("Random Forest Model Evaluation")

    if predictions_filtered.empty:
        st.warning("No testing predictions match the current filters.")
    else:
        evaluation_df = predictions_filtered.copy()
        evaluation_df["residual"] = (
            evaluation_df["resolution_time_hours"]
            - evaluation_df["predicted_resolution_time"]
        )
        evaluation_df["absolute_error"] = evaluation_df["residual"].abs()
        evaluation_df["squared_error"] = evaluation_df["residual"].pow(2)

        mae = float(evaluation_df["absolute_error"].mean())
        rmse = math.sqrt(float(evaluation_df["squared_error"].mean()))
        actual = evaluation_df["resolution_time_hours"].astype(float)
        predicted = evaluation_df["predicted_resolution_time"].astype(float)
        total_variation = float(((actual - actual.mean()) ** 2).sum())
        unexplained_variation = float(((actual - predicted) ** 2).sum())
        r_squared = 1 - unexplained_variation / total_variation

        evaluation_metrics = st.columns(4)
        evaluation_metrics[0].metric("Testing Predictions", f"{len(evaluation_df):,}")
        evaluation_metrics[1].metric("MAE", f"{mae:.2f} hours")
        evaluation_metrics[2].metric("RMSE", f"{rmse:.2f} hours")
        evaluation_metrics[3].metric("R²", f"{r_squared:.4f}")

        model_left, model_right = st.columns(2)

        with model_left:
            actual_predicted_figure = px.scatter(
                evaluation_df,
                x="resolution_time_hours",
                y="predicted_resolution_time",
                color="attack_type",
                opacity=0.65,
                title="Actual vs Predicted Resolution Time",
                labels={
                    "resolution_time_hours": "Actual Time (Hours)",
                    "predicted_resolution_time": "Predicted Time (Hours)",
                    "attack_type": "Attack Type",
                },
            )
            axis_minimum = float(min(actual.min(), predicted.min()))
            axis_maximum = float(max(actual.max(), predicted.max()))
            actual_predicted_figure.add_shape(
                type="line",
                x0=axis_minimum,
                y0=axis_minimum,
                x1=axis_maximum,
                y1=axis_maximum,
                line=dict(color="red", dash="dash"),
            )
            st.plotly_chart(
                finish_figure(actual_predicted_figure),
                use_container_width=True,
            )

        with model_right:
            residual_figure = px.scatter(
                evaluation_df,
                x="predicted_resolution_time",
                y="residual",
                color="attack_type",
                opacity=0.65,
                title="Random Forest Residual Plot",
                labels={
                    "predicted_resolution_time": "Predicted Time (Hours)",
                    "residual": "Residual: Actual − Predicted (Hours)",
                    "attack_type": "Attack Type",
                },
            )
            residual_figure.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(
                finish_figure(residual_figure),
                use_container_width=True,
            )

        distribution_figure = px.histogram(
            evaluation_df,
            x="residual",
            nbins=20,
            title="Random Forest Residual Distribution",
            labels={
                "residual": "Residual (Hours)",
                "count": "Number of Incidents",
            },
            color_discrete_sequence=["#F59E0B"],
        )
        distribution_figure.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(
            finish_figure(distribution_figure),
            use_container_width=True,
        )

        attack_error_df = (
            evaluation_df.groupby("attack_type", as_index=False)
            .agg(
                number_of_incidents=("attack_type", "size"),
                average_actual_time=("resolution_time_hours", "mean"),
                average_predicted_time=("predicted_resolution_time", "mean"),
                mean_residual=("residual", "mean"),
                mae=("absolute_error", "mean"),
                mean_squared_error=("squared_error", "mean"),
            )
        )
        attack_error_df["rmse"] = attack_error_df["mean_squared_error"].pow(0.5)
        attack_error_df = attack_error_df.sort_values("rmse", ascending=False)

        error_long_df = attack_error_df.melt(
            id_vars="attack_type",
            value_vars=["mae", "rmse"],
            var_name="metric",
            value_name="error_hours",
        )

        error_left, error_right = st.columns(2)

        with error_left:
            attack_error_figure = px.bar(
                error_long_df,
                x="attack_type",
                y="error_hours",
                color="metric",
                barmode="group",
                title="Error Magnitude by Attack Type",
                labels={
                    "attack_type": "Attack Type",
                    "error_hours": "Prediction Error (Hours)",
                    "metric": "Metric",
                },
            )
            attack_error_figure.update_xaxes(tickangle=-25)
            st.plotly_chart(
                finish_figure(attack_error_figure),
                use_container_width=True,
            )

        with error_right:
            attack_bias_figure = px.bar(
                attack_error_df,
                x="attack_type",
                y="mean_residual",
                color="mean_residual",
                color_continuous_scale=[
                    [0.0, "#B85C5C"],
                    [0.5, "#E5E7EB"],
                    [1.0, "#4F8A5B"],
                ],
                color_continuous_midpoint=0,
                title="Prediction Bias by Attack Type",
                labels={
                    "attack_type": "Attack Type",
                    "mean_residual": "Mean Residual (Hours)",
                },
            )
            attack_bias_figure.add_hline(y=0, line_dash="dash", line_color="black")
            attack_bias_figure.update_layout(coloraxis_showscale=False)
            attack_bias_figure.update_xaxes(tickangle=-25)
            st.plotly_chart(
                finish_figure(attack_bias_figure),
                use_container_width=True,
            )

        display_error_df = attack_error_df[
            [
                "attack_type",
                "number_of_incidents",
                "average_actual_time",
                "average_predicted_time",
                "mean_residual",
                "mae",
                "rmse",
            ]
        ].round(2)
        st.dataframe(display_error_df, use_container_width=True, hide_index=True)
        st.info(
            "Positive residuals indicate under-prediction. Negative residuals "
            "indicate over-prediction."
        )


with prediction_tab:
    st.subheader("Predict Incident Resolution Time")
    st.write(
        "Enter a new incident. The dashboard sends the values to the FastAPI "
        "endpoint deployed on Render and displays the Random Forest prediction."
    )
    st.info(
        "The free Render service can sleep after inactivity. The first request "
        "may take about one minute while the service wakes up."
    )

    def category_values(column_name):
        return sorted(prepared_df[column_name].dropna().astype(str).unique().tolist())

    with st.form("prediction_form"):
        form_left, form_right = st.columns(2)

        with form_left:
            country = st.selectbox("Country", category_values("country"))
            year = st.selectbox(
                "Year",
                list(range(maximum_year, minimum_year - 1, -1)),
            )
            attack_type = st.selectbox(
                "Attack Type",
                category_values("attack_type"),
            )
            target_industry = st.selectbox(
                "Target Industry",
                category_values("target_industry"),
            )
            financial_loss = st.number_input(
                "Financial Loss (Million USD)",
                min_value=0.0,
                max_value=1000.0,
                value=50.0,
                step=1.0,
            )

        with form_right:
            affected_users = st.number_input(
                "Number of Affected Users",
                min_value=1,
                max_value=100_000_000,
                value=500_000,
                step=1_000,
            )
            attack_source = st.selectbox(
                "Attack Source",
                category_values("attack_source"),
            )
            vulnerability_type = st.selectbox(
                "Vulnerability Type",
                category_values("vulnerability_type"),
            )
            defense_mechanism = st.selectbox(
                "Defense Mechanism",
                category_values("defense_mechanism"),
            )

        submitted = st.form_submit_button("Predict Resolution Time")

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

        try:
            with st.spinner("Sending the incident to the deployed model..."):
                response = requests.post(
                    f"{API_URL}/predict",
                    json=payload,
                    timeout=120,
                )

            if response.status_code != 200:
                try:
                    detail = response.json()
                except ValueError:
                    detail = response.text
                st.error(
                    f"Prediction failed with status {response.status_code}: {detail}"
                )
            else:
                result = response.json()
                predicted_hours = float(result["predicted_resolution_time_hours"])
                predicted_days = float(result["predicted_resolution_time_days"])

                st.success("Prediction completed successfully.")
                result_left, result_right = st.columns(2)
                result_left.metric(
                    "Predicted Resolution Time",
                    f"{predicted_hours:.2f} hours",
                )
                result_right.metric(
                    "Equivalent Duration",
                    f"{predicted_days:.2f} days",
                )

                with st.expander("View submitted data and API response"):
                    st.write("Submitted incident")
                    st.json(payload)
                    st.write("API response")
                    st.json(result)

        except requests.Timeout:
            st.error(
                "The request timed out. The Render service may still be waking "
                "up. Wait one minute and try again."
            )
        except requests.ConnectionError:
            st.error(
                "The dashboard could not connect to the FastAPI service. "
                "Confirm that the Render deployment is live."
            )
        except (KeyError, TypeError, ValueError) as error:
            st.error(f"The API returned an unexpected response: {error}")
        except requests.RequestException as error:
            st.error(f"The prediction request failed: {error}")


st.divider()
st.caption(
    "Part II: Streamlit dashboard connected to a FastAPI endpoint on Render. "
    "Part I data preparation, clustering, and model training used Apache Spark."
)