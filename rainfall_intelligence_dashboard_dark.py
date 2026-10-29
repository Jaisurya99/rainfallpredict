import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Rainfall Intelligence Dashboard",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme styling
st.markdown("""
<style>
:root { color-scheme: dark; }
[data-testid="stAppViewContainer"] {
    background: #0b1120;
    color: #e5edf7;
}
[data-testid="stHeader"] { background: rgba(11,17,32,0.92); }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080e1b 0%, #14263d 100%);
    border-right: 1px solid #26364d;
}
[data-testid="stSidebar"] * { color: #e5edf7 !important; }
.block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1450px; }
.hero {
    padding: 1.6rem 1.8rem; border-radius: 20px; color: white;
    background: linear-gradient(115deg, #111827 0%, #164e63 58%, #0f766e 100%);
    margin-bottom: 1.2rem; border: 1px solid #28566b;
    box-shadow: 0 10px 28px rgba(0,0,0,.25);
}
.hero h1 { color: #f8fafc; font-size: 2.1rem; margin: 0 0 .35rem 0; }
.hero p { color: #c8e7f0; font-size: 1rem; margin: 0; }
.section-card { background: #111c2e; border: 1px solid #26364d; border-radius: 16px; padding: 1rem 1.2rem; }
[data-testid="stMetric"] {
    background: #111c2e; border: 1px solid #26364d; padding: 1rem 1.1rem;
    border-radius: 15px; box-shadow: 0 4px 14px rgba(0,0,0,.16);
}
[data-testid="stMetricLabel"] { color: #a9bbd0; }
[data-testid="stMetricValue"] { color: #f1f5f9; }
[data-testid="stMarkdown"], [data-testid="stText"], p, label, li { color: #dce6f2; }
h1, h2, h3, h4, h5, h6 { color: #f1f5f9; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: #172438; color: #cbd5e1; border-radius: 10px 10px 0 0;
    padding: 10px 16px;
}
.stTabs [aria-selected="true"] {
    background: #111c2e !important; color: #67e8f9 !important;
}
.stButton button, .stDownloadButton button {
    border-radius: 10px; font-weight: 600; background: #164e63;
    color: #f8fafc; border: 1px solid #28758a;
}
.stButton button:hover, .stDownloadButton button:hover {
    background: #0f766e; color: white; border-color: #5eead4;
}
div[data-testid="stAlert"] { border-radius: 12px; }
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border: 1px solid #26364d; border-radius: 10px; overflow: hidden;
}
input, textarea {
    background-color: #111c2e !important; color: #f1f5f9 !important;
}
[data-baseweb="select"] > div {
    background-color: #111c2e; color: #f1f5f9; border-color: #34465e;
}
hr { border-color: #26364d; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Keep Matplotlib charts consistent with the dashboard's dark theme.
plt.rcParams.update({
    "figure.facecolor": "#111c2e",
    "axes.facecolor": "#111c2e",
    "axes.edgecolor": "#94a3b8",
    "axes.labelcolor": "#e2e8f0",
    "text.color": "#e2e8f0",
    "xtick.color": "#cbd5e1",
    "ytick.color": "#cbd5e1",
    "grid.color": "#334155",
    "savefig.facecolor": "#111c2e",
})

st.markdown("""
<div class="hero">
  <h1>🌧️ Rainfall Intelligence Dashboard</h1>
  <p>Historical rainfall exploration · Regional comparisons · Machine learning evaluation · Next-year estimates</p>
</div>
""", unsafe_allow_html=True)
st.caption("CA2 • Machine Learning Project | Indian subdivision rainfall data, 1901–2017")

# Locate the dataset beside this Python file.
BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_FOLDER, "Sub_Division_IMD_2017.csv")


@st.cache_data
def load_data(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.upper()

    if "SUBDIVISION" not in df.columns or "YEAR" not in df.columns or "ANNUAL" not in df.columns:
        raise ValueError("Dataset must contain SUBDIVISION, YEAR and ANNUAL columns.")

    df["SUBDIVISION"] = df["SUBDIVISION"].astype("string").str.strip()

    for col in df.columns:
        if col != "SUBDIVISION":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["SUBDIVISION", "YEAR"])
    df["YEAR"] = df["YEAR"].astype(int)
    df = df.drop_duplicates()
    df = df.sort_values(["SUBDIVISION", "YEAR"]).reset_index(drop=True)
    return df


@st.cache_data
def prepare_model_data(df):
    work = df.sort_values(["SUBDIVISION", "YEAR"]).copy()
    group = work.groupby("SUBDIVISION", sort=False)

    work["lag_0"] = work["ANNUAL"]
    work["lag_1"] = group["ANNUAL"].shift(1)
    work["lag_2"] = group["ANNUAL"].shift(2)
    work["previous_year"] = group["YEAR"].shift(1)
    work["two_years_before"] = group["YEAR"].shift(2)
    work["target_year"] = group["YEAR"].shift(-1)
    work["target"] = group["ANNUAL"].shift(-1)
    work["rolling_mean"] = group["ANNUAL"].transform(
        lambda values: values.rolling(3).mean()
    )

    features = ["lag_0", "lag_1", "lag_2", "rolling_mean"]
    usable = work[
        (work["target_year"] == work["YEAR"] + 1)
        & (work["previous_year"] == work["YEAR"] - 1)
        & (work["two_years_before"] == work["YEAR"] - 2)
    ].copy()
    usable = usable.dropna(subset=features + ["target"])

    if usable.empty:
        return usable, features

    return usable, features


@st.cache_resource
def train_models(_model_data, features):
    model_data = _model_data.copy()
    train = model_data[model_data["target_year"] <= 2007]
    test = model_data[model_data["target_year"] > 2007]

    if train.empty or test.empty:
        return None, None, None, None, None

    X_train = train[features]
    y_train = train["target"]
    X_test = test[features]
    y_test = test["target"]

    models = {
        "Linear Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=200, min_samples_leaf=3,
                random_state=42, n_jobs=-1
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.05,
                max_depth=2, random_state=42
            ))
        ])
    }

    rows = []
    predictions = {}

    baseline_pred = test["lag_0"]
    rows.append({
        "Model": "Previous-Year Baseline",
        "MAE": mean_absolute_error(y_test, baseline_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, baseline_pred)),
        "R²": r2_score(y_test, baseline_pred),
        "Test samples": len(test)
    })

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[name] = pred
        rows.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R²": r2_score(y_test, pred),
            "Test samples": len(test)
        })

    metrics = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    best_name = metrics[metrics["Model"] != "Previous-Year Baseline"].sort_values("MAE").iloc[0]["Model"]

    test_results = test[["SUBDIVISION", "YEAR", "target_year", "target"]].copy()
    test_results = test_results.rename(columns={
        "target_year": "Prediction Year",
        "target": "Actual Rainfall (mm)"
    })
    for name, pred in predictions.items():
        test_results[name + " Prediction (mm)"] = pred

    return models, metrics, test_results, best_name, (X_train, y_train, X_test, y_test)


if not os.path.exists(DATA_FILE):
    st.error(
        "Dataset file not found. Put Sub_Division_IMD_2017.csv in the same folder "
        "as streamlit_app.py."
    )
    st.stop()

try:
    data = load_data(DATA_FILE)
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()

annual = data.dropna(subset=["ANNUAL"]).copy()
subdivisions = sorted(annual["SUBDIVISION"].dropna().unique().tolist())

st.sidebar.markdown("## 🌧️ Rainfall Lab")
st.sidebar.caption("Explore the historical dataset")
st.sidebar.divider()
st.sidebar.header("Dashboard filters")
chosen_subdivision = st.sidebar.selectbox(
    "Choose a subdivision",
    ["All subdivisions"] + subdivisions
)

min_year = int(annual["YEAR"].min())
max_year = int(annual["YEAR"].max())
year_range = st.sidebar.slider(
    "Choose year range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

filtered = annual[
    annual["YEAR"].between(year_range[0], year_range[1])
].copy()

if chosen_subdivision != "All subdivisions":
    filtered = filtered[filtered["SUBDIVISION"] == chosen_subdivision]

# Summary cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Dataset records", f"{len(filtered):,}")
c2.metric("Subdivisions shown", f"{filtered['SUBDIVISION'].nunique():,}")
c3.metric("Average annual rainfall", f"{filtered['ANNUAL'].mean():,.1f} mm" if not filtered.empty else "N/A")
c4.metric("Missing annual values", f"{data['ANNUAL'].isna().sum():,}")

st.info(
    "Data note: the dataset ends in 2017. The next-year estimate is a historical-pattern estimate, "
    "not live weather data or an official meteorological forecast."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview & EDA", "🗺️ Regional analysis", "🤖 Model performance", "🔮 Prediction explorer"
])

with tab1:
    st.subheader("Historical rainfall overview")
    left, right = st.columns(2)

    with left:
        st.markdown("**Annual rainfall distribution**")
        fig, ax = plt.subplots(figsize=(7, 4))
        if not filtered.empty:
            ax.hist(filtered["ANNUAL"].dropna(), bins=30, edgecolor="#cbd5e1")
        ax.set_xlabel("Annual rainfall (mm)")
        ax.set_ylabel("Number of observations")
        ax.set_title("Annual rainfall distribution")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown("**Average annual rainfall by year**")
        yearly = filtered.groupby("YEAR")["ANNUAL"].mean()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(yearly.index, yearly.values, marker=".", linewidth=2)
        ax.set_xlabel("Year")
        ax.set_ylabel("Average rainfall (mm)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    month_names = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                   "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    available_months = [m for m in month_names if m in data.columns]

    if available_months:
        st.markdown("**Average monthly rainfall**")
        monthly = data[available_months].mean()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(monthly.index, monthly.values)
        ax.set_ylabel("Average rainfall (mm)")
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("**Monthly rainfall correlation**")
        corr = data[available_months].corr()
        fig, ax = plt.subplots(figsize=(9, 6))
        image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(available_months)), available_months, rotation=45)
        ax.set_yticks(range(len(available_months)), available_months)
        fig.colorbar(image, ax=ax, label="Correlation")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("**Filtered data preview**")
    st.dataframe(filtered.head(100), use_container_width=True)

with tab2:
    st.subheader("Compare subdivisions")
    if chosen_subdivision == "All subdivisions":
        region_data = filtered
    else:
        region_data = annual[annual["YEAR"].between(year_range[0], year_range[1])]

    if not region_data.empty:
        regional = region_data.groupby("SUBDIVISION")["ANNUAL"].mean().sort_values(ascending=False)
        st.markdown("**Average annual rainfall by subdivision**")
        st.bar_chart(regional)
        st.markdown("**Summary statistics**")
        st.dataframe(
            region_data.groupby("SUBDIVISION")["ANNUAL"].agg(
                ["count", "mean", "median", "std", "min", "max"]
            ).round(2),
            use_container_width=True
        )
    else:
        st.warning("No records match the selected filters.")

with tab3:
    st.subheader("Machine learning model comparison")
    model_data, features = prepare_model_data(data)

    if model_data.empty:
        st.warning("There are not enough consecutive yearly records to train the models.")
    else:
        with st.spinner("Training models on the historical dataset..."):
            models, metrics, test_results, best_name, split_data = train_models(model_data, features)

        if metrics is None:
            st.warning("Could not create the chronological training/test split.")
        else:
            best_row = metrics[metrics["Model"] == best_name].iloc[0]
            a, b, c = st.columns(3)
            a.metric("Best ML model by MAE", best_name)
            b.metric("Best model MAE", f"{best_row['MAE']:.2f} mm")
            c.metric("Best model RMSE", f"{best_row['RMSE']:.2f} mm")

            st.dataframe(metrics.round(3), use_container_width=True)
            st.markdown("**MAE comparison (lower is better)**")
            st.bar_chart(metrics.set_index("Model")[["MAE"]])

            st.caption(
                "Training uses target years up to 2007; testing uses later target years. "
                "The baseline assumes next year's rainfall equals the current year's rainfall."
            )
            st.download_button(
                "Download model metrics CSV",
                metrics.to_csv(index=False).encode("utf-8"),
                file_name="model_metrics.csv",
                mime="text/csv"
            )
            st.download_button(
                "Download test predictions CSV",
                test_results.to_csv(index=False).encode("utf-8"),
                file_name="test_predictions.csv",
                mime="text/csv"
            )

with tab4:
    st.subheader("Future Rainfall Prediction")
    st.write(
        "Select a subdivision to estimate annual rainfall for the year immediately "
        "after its latest available three consecutive years of records."
    )
    model_data, features = prepare_model_data(data)

    if model_data.empty:
        st.warning("Not enough consecutive yearly records to train a model.")
    else:
        models, metrics, test_results, best_name, split_data = train_models(model_data, features)

        if models is None:
            st.warning("Models could not be trained.")
        else:
            selected_model_name = st.selectbox(
                "Choose prediction model",
                ["Linear Regression", "Random Forest", "Gradient Boosting"],
                index=["Linear Regression", "Random Forest", "Gradient Boosting"].index(best_name)
            )
            selected_region = st.selectbox("Choose subdivision for prediction", subdivisions)

            # Train the selected model on all available supervised historical examples.
            future_model = models[selected_model_name]
            future_model.fit(model_data[features], model_data["target"])

            # Build a downloadable next-year estimate table for every subdivision.
            all_estimates = []
            for subdivision_name in subdivisions:
                part = data[data["SUBDIVISION"] == subdivision_name].sort_values("YEAR")
                part = part.dropna(subset=["ANNUAL"])
                if len(part) < 3:
                    continue

                latest_three = part.tail(3)
                recent_years = latest_three["YEAR"].to_numpy()
                recent_values = latest_three["ANNUAL"].to_numpy(dtype=float)

                if (recent_years[1] != recent_years[0] + 1 or
                        recent_years[2] != recent_years[1] + 1):
                    continue

                prediction_input = pd.DataFrame([{
                    "lag_0": recent_values[-1],
                    "lag_1": recent_values[-2],
                    "lag_2": recent_values[-3],
                    "rolling_mean": float(np.mean(recent_values))
                }], columns=features)
                predicted_rainfall = max(0.0, float(future_model.predict(prediction_input)[0]))
                all_estimates.append({
                    "SUBDIVISION": subdivision_name,
                    "LATEST_DATA_YEAR": int(recent_years[-1]),
                    "ESTIMATE_YEAR": int(recent_years[-1]) + 1,
                    "LATEST_ANNUAL_RAINFALL_MM": round(float(recent_values[-1]), 2),
                    "ESTIMATED_ANNUAL_RAINFALL_MM": round(predicted_rainfall, 2),
                    "MODEL": selected_model_name
                })

            st.markdown("### Download next-year estimates")
            if all_estimates:
                estimates_df = pd.DataFrame(all_estimates).sort_values("SUBDIVISION")
                st.write(
                    "This table contains one next-year estimate per subdivision where the latest "
                    "three usable annual records are consecutive. The estimate year depends on the "
                    "latest data year available for that subdivision."
                )
                st.dataframe(estimates_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Download all next-year rainfall estimates (CSV)",
                    data=estimates_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="next_year_estimates.csv",
                    mime="text/csv",
                    key="download_all_next_year_estimates"
                )
            else:
                st.warning("No subdivisions have three consecutive usable annual records for estimates.")

            region = data[data["SUBDIVISION"] == selected_region].sort_values("YEAR")
            region = region.dropna(subset=["ANNUAL"])

            if len(region) < 3:
                st.warning("This subdivision has fewer than three usable annual rainfall records.")
            else:
                last_three = region.tail(3)
                years = last_three["YEAR"].to_numpy()
                values = last_three["ANNUAL"].to_numpy()

                if years[1] != years[0] + 1 or years[2] != years[1] + 1:
                    st.warning("The latest three usable records are not consecutive years, so an estimate is not shown.")
                else:
                    input_row = pd.DataFrame([{
                        "lag_0": values[-1],
                        "lag_1": values[-2],
                        "lag_2": values[-3],
                        "rolling_mean": np.mean(values)
                    }], columns=features)

                    estimate = max(0, float(future_model.predict(input_row)[0]))
                    forecast_year = int(years[-1]) + 1

                    st.success("Prediction generated from historical rainfall data")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Subdivision", selected_region)
                    col2.metric("Estimate year", str(forecast_year))
                    col3.metric("Estimated annual rainfall", f"{estimate:,.1f} mm")

                    st.write(
                        f"Latest observed rainfall: **{values[-1]:,.1f} mm in {int(years[-1])}**."
                    )
                    st.markdown("**Recent historical rainfall**")
                    history = region[["YEAR", "ANNUAL"]].tail(20).copy()
                    st.line_chart(history.set_index("YEAR"))

                    st.caption(
                        "The dataset ends in 2017. If 2017 is the latest record, this estimates 2018, "
                        "not current 2026 rainfall. To estimate 2026 meaningfully, you need newer data "
                        "and further validation. This is a historical-pattern estimate, not an official forecast."
                    )

st.divider()
st.caption("Rainfall Prediction Using Machine Learning • Student CA2 project")
st.markdown("---")
st.caption("Built with Python • Pandas • Matplotlib • Scikit-learn • Streamlit | Educational demonstration only")
