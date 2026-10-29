
# ==========================================================
# RAINFALL PREDICTION USING MACHINE LEARNING
# Dataset: Sub_Division_IMD_2017.csv
# ==========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------------------------------------
# 1. SET PROJECT PATHS
# ----------------------------------------------------------

folder = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(folder, "Sub_Division_IMD_2017.csv")
output_folder = os.path.join(folder, "outputs")
os.makedirs(output_folder, exist_ok=True)

print("=" * 55)
print("RAINFALL PREDICTION USING MACHINE LEARNING")
print("=" * 55)

# ----------------------------------------------------------
# 2. LOAD DATASET
# ----------------------------------------------------------

if not os.path.exists(file_path):
    raise FileNotFoundError(
        "Place Sub_Division_IMD_2017.csv in the same folder "
        "as this Python file."
    )

data = pd.read_csv(file_path, encoding="utf-8-sig")
data.columns = data.columns.str.strip().str.upper()

required = ["SUBDIVISION", "YEAR", "ANNUAL"]
for column in required:
    if column not in data.columns:
        raise ValueError("Missing required column: " + column)

print("\nOriginal dataset shape:", data.shape)
print("\nFirst five rows:")
print(data.head())

# ----------------------------------------------------------
# 3. DATA CLEANING
# ----------------------------------------------------------

data["SUBDIVISION"] = data["SUBDIVISION"].astype("string").str.strip()

numeric_columns = [
    col for col in data.columns if col != "SUBDIVISION"
]

for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.dropna(subset=["SUBDIVISION", "YEAR"])
data["YEAR"] = data["YEAR"].astype(int)

duplicates = data.duplicated().sum()
data = data.drop_duplicates()

data = data.sort_values(
    ["SUBDIVISION", "YEAR"]
).reset_index(drop=True)

print("\nDuplicate rows removed:", duplicates)
print("Cleaned dataset shape:", data.shape)

missing_report = data.isnull().sum().reset_index()
missing_report.columns = ["Column", "Missing_Values"]
missing_report.to_csv(
    os.path.join(output_folder, "missing_values_report.csv"),
    index=False
)

print("\nMissing values by column:")
print(missing_report.to_string(index=False))

data.to_csv(
    os.path.join(output_folder, "cleaned_data.csv"),
    index=False
)

# Keep genuine zero-rainfall observations.
# Missing target values are not replaced with zero.

# ----------------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# ----------------------------------------------------------

annual_data = data.dropna(subset=["ANNUAL"]).copy()

print("\nAnnual rainfall statistics:")
print(annual_data["ANNUAL"].describe())

print("\nNumber of subdivisions:", data["SUBDIVISION"].nunique())
print("Year range:", data["YEAR"].min(), "-", data["YEAR"].max())

# Graph 1: Annual rainfall distribution
plt.figure(figsize=(9, 5))
plt.hist(annual_data["ANNUAL"], bins=30, edgecolor="black")
plt.title("Annual Rainfall Distribution")
plt.xlabel("Annual Rainfall (mm)")
plt.ylabel("Number of Observations")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "01_distribution.png"))
plt.show()

# Graph 2: Average annual rainfall over time
yearly_average = annual_data.groupby("YEAR")["ANNUAL"].mean()

plt.figure(figsize=(11, 5))
plt.plot(yearly_average.index, yearly_average.values)
plt.title("Historical Average Annual Rainfall")
plt.xlabel("Year")
plt.ylabel("Average Rainfall (mm)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "02_yearly_trend.png"))
plt.show()

# Graph 3: Selected subdivision trends
plt.figure(figsize=(11, 6))

selected = annual_data["SUBDIVISION"].drop_duplicates().head(5)

for subdivision in selected:
    part = annual_data[
        annual_data["SUBDIVISION"] == subdivision
    ].sort_values("YEAR")

    plt.plot(part["YEAR"], part["ANNUAL"], label=subdivision)

plt.title("Rainfall Trends by Subdivision")
plt.xlabel("Year")
plt.ylabel("Annual Rainfall (mm)")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "03_subdivision_trends.png"))
plt.show()

# Graph 4: Average monthly rainfall
months = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"
]
months = [month for month in months if month in data.columns]

if months:
    monthly_average = data[months].mean()

    plt.figure(figsize=(10, 5))
    plt.bar(monthly_average.index, monthly_average.values)
    plt.title("Average Monthly Rainfall")
    plt.xlabel("Month")
    plt.ylabel("Average Rainfall (mm)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "04_monthly_rainfall.png"))
    plt.show()

# Graph 5: Monthly rainfall correlation
if len(months) >= 2:
    correlation = data[months].corr()

    plt.figure(figsize=(9, 7))
    plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(months)), months, rotation=45)
    plt.yticks(range(len(months)), months)
    plt.title("Monthly Rainfall Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "05_correlation.png"))
    plt.show()

    correlation.to_csv(
        os.path.join(output_folder, "monthly_correlation.csv")
    )

# ----------------------------------------------------------
# 5. STATISTICAL ANALYSIS AND OUTLIERS
# ----------------------------------------------------------

print("\nStatistical analysis")
print("Mean:", round(annual_data["ANNUAL"].mean(), 2), "mm")
print("Median:", round(annual_data["ANNUAL"].median(), 2), "mm")
print("Standard deviation:", round(annual_data["ANNUAL"].std(), 2), "mm")

statistics = annual_data.groupby("SUBDIVISION")["ANNUAL"].agg(
    ["count", "mean", "median", "std", "min", "max"]
)

statistics.to_csv(
    os.path.join(output_folder, "subdivision_statistics.csv")
)

q1 = annual_data["ANNUAL"].quantile(0.25)
q3 = annual_data["ANNUAL"].quantile(0.75)
iqr = q3 - q1

outliers = annual_data[
    (annual_data["ANNUAL"] < q1 - 1.5 * iqr) |
    (annual_data["ANNUAL"] > q3 + 1.5 * iqr)
]

print("Potential outlier observations:", len(outliers))

# Extreme rainfall may be genuine, so we flag rather than delete it.
outliers.to_csv(
    os.path.join(output_folder, "potential_outliers.csv"),
    index=False
)

# ----------------------------------------------------------
# 6. FEATURE ENGINEERING
# ----------------------------------------------------------

data = data.sort_values(
    ["SUBDIVISION", "YEAR"]
).reset_index(drop=True)

group = data.groupby("SUBDIVISION", sort=False)

# Inputs use only rainfall values from past years.
data["lag_0"] = data["ANNUAL"]
data["lag_1"] = group["ANNUAL"].shift(1)
data["lag_2"] = group["ANNUAL"].shift(2)

data["year_lag_1"] = group["YEAR"].shift(1)
data["year_lag_2"] = group["YEAR"].shift(2)

data["target_year"] = group["YEAR"].shift(-1)
data["target"] = group["ANNUAL"].shift(-1)

data["rolling_mean"] = group["ANNUAL"].transform(
    lambda values: values.rolling(3).mean()
)

# Keep only consecutive years to avoid treating a missing
# calendar year as the next year.
model_data = data[
    (data["target_year"] == data["YEAR"] + 1) &
    (data["year_lag_1"] == data["YEAR"] - 1) &
    (data["year_lag_2"] == data["YEAR"] - 2)
].copy()

features = ["lag_0", "lag_1", "lag_2", "rolling_mean"]

model_data = model_data.dropna(subset=features + ["target"])

if model_data.empty:
    raise ValueError("No usable training rows were found.")

print("\nRows available for modeling:", len(model_data))

# ----------------------------------------------------------
# 7. CHRONOLOGICAL TRAINING AND TESTING
# ----------------------------------------------------------

# Earlier years train the models.
# Later years evaluate predictions on unseen future periods.

train = model_data[model_data["target_year"] <= 2007]
test = model_data[model_data["target_year"] > 2007]

if train.empty or test.empty:
    raise ValueError("Training or testing data is empty.")

X_train = train[features]
y_train = train["target"]

X_test = test[features]
y_test = test["target"]

print("Training rows:", len(train))
print("Testing rows:", len(test))
print(
    "Test target years:",
    int(test["target_year"].min()),
    "to",
    int(test["target_year"].max())
)

# ----------------------------------------------------------
# 8. TRAIN MACHINE LEARNING MODELS
# ----------------------------------------------------------

models = {
    "Linear Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "Gradient Boosting": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            random_state=42
        ))
    ])
}

predictions = {}
results = []

# Baseline: next year's rainfall equals the current year's.
baseline_rows = test.dropna(subset=["lag_0"])
baseline_actual = baseline_rows["target"]
baseline_pred = baseline_rows["lag_0"]

results.append({
    "Model": "Previous-Year Baseline",
    "MAE": mean_absolute_error(baseline_actual, baseline_pred),
    "RMSE": np.sqrt(mean_squared_error(baseline_actual, baseline_pred)),
    "R2": r2_score(baseline_actual, baseline_pred),
    "Test_Samples": len(baseline_rows)
})

for name, model in models.items():
    print("Training", name)
    model.fit(X_train, y_train)

    predicted = model.predict(X_test)
    predictions[name] = predicted

    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, predicted),
        "RMSE": np.sqrt(mean_squared_error(y_test, predicted)),
        "R2": r2_score(y_test, predicted),
        "Test_Samples": len(y_test)
    })

# ----------------------------------------------------------
# 9. EVALUATE AND COMPARE MODELS
# ----------------------------------------------------------

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("MAE")

print("\nModel performance:")
print(results_df.to_string(index=False))

results_df.to_csv(
    os.path.join(output_folder, "model_metrics.csv"),
    index=False
)

# Graph 6: Model comparison
plt.figure(figsize=(9, 5))
plt.bar(results_df["Model"], results_df["MAE"])
plt.title("Comparison of Models Using MAE")
plt.xlabel("Model")
plt.ylabel("Mean Absolute Error (mm)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "06_model_comparison.png"))
plt.show()

# Select best ML model by MAE, excluding the baseline.
ml_results = results_df[
    results_df["Model"] != "Previous-Year Baseline"
]

best_name = ml_results.iloc[0]["Model"]
best_model = models[best_name]
best_predictions = predictions[best_name]

print("\nBest ML model based on MAE:", best_name)

# Graph 7: Actual vs predicted rainfall
plt.figure(figsize=(7, 7))
plt.scatter(y_test, best_predictions, alpha=0.6)

low = min(float(y_test.min()), float(best_predictions.min()))
high = max(float(y_test.max()), float(best_predictions.max()))

plt.plot([low, high], [low, high])
plt.title("Actual vs Predicted Rainfall")
plt.xlabel("Actual Rainfall (mm)")
plt.ylabel("Predicted Rainfall (mm)")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "07_actual_vs_predicted.png"))
plt.show()

# Save all test predictions
test_results = test[
    ["SUBDIVISION", "YEAR", "target_year", "target"]
].copy()

test_results = test_results.rename(
    columns={"target": "Actual_Rainfall_mm"}
)

for name, values in predictions.items():
    safe_name = name.lower().replace(" ", "_")
    test_results[safe_name + "_prediction_mm"] = values

test_results.to_csv(
    os.path.join(output_folder, "test_predictions.csv"),
    index=False
)

# ----------------------------------------------------------
# 10. PREDICT NEXT AVAILABLE YEAR
# ----------------------------------------------------------

# The data ends in 2017, so the following values are
# historical-pattern estimates, not current weather forecasts.

forecast_rows = []

for subdivision, part in data.groupby("SUBDIVISION"):
    part = part.sort_values("YEAR").dropna(subset=["ANNUAL"])

    if len(part) < 3:
        continue

    last_three = part.tail(3)
    years = last_three["YEAR"].to_numpy()

    if not (
        years[1] == years[0] + 1 and
        years[2] == years[1] + 1
    ):
        continue

    values = last_three["ANNUAL"].to_numpy()

    future_input = pd.DataFrame([{
        "lag_0": values[-1],
        "lag_1": values[-2],
        "lag_2": values[-3],
        "rolling_mean": np.mean(values)
    }])

    estimate = best_model.predict(future_input[features])[0]

    forecast_rows.append({
        "SUBDIVISION": subdivision,
        "Last_Observed_Year": int(years[-1]),
        "Forecast_Year": int(years[-1]) + 1,
        "Estimated_Rainfall_mm": max(0, estimate)
    })

forecast_df = pd.DataFrame(forecast_rows)

if not forecast_df.empty:
    forecast_df.to_csv(
        os.path.join(output_folder, "next_year_estimates.csv"),
        index=False
    )

    print("\nNext-year historical-pattern estimates:")
    print(forecast_df.to_string(index=False))
else:
    print("\nNo suitable subdivisions found for forecasting.")

# ----------------------------------------------------------
# 11. SAVE FINAL SUMMARY
# ----------------------------------------------------------

with open(
    os.path.join(output_folder, "project_summary.txt"),
    "w",
    encoding="utf-8"
) as file:
    file.write("Rainfall Prediction Using Machine Learning\n")
    file.write("Best ML model: " + best_name + "\n")
    file.write("Training rows: " + str(len(train)) + "\n")
    file.write("Testing rows: " + str(len(test)) + "\n")
    file.write("\nModel comparison:\n")
    file.write(results_df.to_string(index=False))

print("\n" + "=" * 55)
print("PROJECT COMPLETED SUCCESSFULLY")
print("Results and graphs are saved in:", output_folder)
print("=" * 55)
