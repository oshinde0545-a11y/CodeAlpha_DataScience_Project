# ============================================================
# TASK 2: UNEMPLOYMENT ANALYSIS WITH PYTHON
# Google Colab - Fully Runnable Program
# ============================================================

# -------------------------------
# 1. IMPORT LIBRARIES
# -------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("       UNEMPLOYMENT ANALYSIS")
print("=" * 60)


# -------------------------------
# 2. CREATE SAMPLE DATASET
# -------------------------------
# This makes the program directly runnable in Google Colab.
# You can later replace this section with your own CSV file.

dates = pd.date_range(
    start="2019-01-01",
    end="2022-12-01",
    freq="MS"
)

np.random.seed(42)

unemployment = []

for date in dates:

    # Normal unemployment before COVID
    if date < pd.Timestamp("2020-03-01"):
        rate = np.random.uniform(5, 9)

    # Higher unemployment during COVID
    elif date <= pd.Timestamp("2021-12-01"):
        rate = np.random.uniform(8, 18)

        # Very high unemployment during early COVID
        if date.year == 2020 and date.month in [4, 5, 6]:
            rate = np.random.uniform(15, 25)

    # Recovery after COVID
    else:
        rate = np.random.uniform(5, 9)

    unemployment.append(round(rate, 2))


df = pd.DataFrame({
    "Date": dates,
    "Region": np.random.choice(
        ["North", "South", "East", "West"],
        len(dates)
    ),
    "Estimated Unemployment Rate (%)": unemployment
})


# -------------------------------
# 3. DISPLAY DATASET
# -------------------------------

print("\nFirst 5 rows of dataset:")
print(df.head())

print("\nLast 5 rows of dataset:")
print(df.tail())

print("\nDataset shape:")
print(df.shape)


# -------------------------------
# 4. DATA CLEANING
# -------------------------------

df.columns = df.columns.str.strip()

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

df["Estimated Unemployment Rate (%)"] = pd.to_numeric(
    df["Estimated Unemployment Rate (%)"],
    errors="coerce"
)

df = df.drop_duplicates()

df = df.dropna(
    subset=[
        "Date",
        "Estimated Unemployment Rate (%)"
    ]
)

df = df.sort_values("Date")


print("\nMissing values:")
print(df.isnull().sum())


# -------------------------------
# 5. STATISTICAL ANALYSIS
# -------------------------------

rate_column = "Estimated Unemployment Rate (%)"

print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)

print(df[rate_column].describe())

print(
    "\nAverage Unemployment Rate:",
    round(df[rate_column].mean(), 2),
    "%"
)

print(
    "Minimum Unemployment Rate:",
    round(df[rate_column].min(), 2),
    "%"
)

print(
    "Maximum Unemployment Rate:",
    round(df[rate_column].max(), 2),
    "%"
)


# -------------------------------
# 6. OVERALL UNEMPLOYMENT TREND
# -------------------------------

monthly_unemployment = (
    df.groupby("Date")[rate_column]
    .mean()
)

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_unemployment.index,
    monthly_unemployment.values,
    marker="o"
)

plt.title(
    "Unemployment Rate Trend Over Time"
)

plt.xlabel("Date")

plt.ylabel(
    "Unemployment Rate (%)"
)

plt.grid(True)

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


# -------------------------------
# 7. COVID-19 IMPACT ANALYSIS
# -------------------------------

df["Period"] = np.where(
    df["Date"] < pd.Timestamp("2020-03-01"),
    "Pre-COVID",
    np.where(
        df["Date"] <= pd.Timestamp("2021-12-31"),
        "COVID Period",
        "Post-COVID"
    )
)

covid_analysis = (
    df.groupby("Period")[rate_column]
    .mean()
)

print("\n" + "=" * 60)
print("COVID-19 IMPACT ANALYSIS")
print("=" * 60)

print(
    covid_analysis.round(2)
)


# -------------------------------
# 8. COVID BAR CHART
# -------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    covid_analysis.index,
    covid_analysis.values
)

plt.title(
    "Average Unemployment Rate:"
    " Pre-COVID vs COVID vs Post-COVID"
)

plt.xlabel("Period")

plt.ylabel(
    "Average Unemployment Rate (%)"
)

plt.tight_layout()

plt.show()


# -------------------------------
# 9. HIGHEST COVID UNEMPLOYMENT
# -------------------------------

covid_data = df[
    (df["Date"] >= pd.Timestamp("2020-03-01"))
    &
    (df["Date"] <= pd.Timestamp("2021-12-31"))
]

highest_covid = covid_data.loc[
    covid_data[rate_column].idxmax()
]

print("\n" + "=" * 60)
print("HIGHEST UNEMPLOYMENT DURING COVID")
print("=" * 60)

print(
    "Date:",
    highest_covid["Date"].strftime("%Y-%m-%d")
)

print(
    "Unemployment Rate:",
    round(
        highest_covid[rate_column],
        2
    ),
    "%"
)


# -------------------------------
# 10. REGIONAL ANALYSIS
# -------------------------------

region_average = (
    df.groupby("Region")[rate_column]
    .mean()
    .sort_values(ascending=False)
)

print("\n" + "=" * 60)
print("REGIONAL ANALYSIS")
print("=" * 60)

print(
    region_average.round(2)
)


# -------------------------------
# 11. REGIONAL BAR CHART
# -------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    region_average.index,
    region_average.values
)

plt.title(
    "Average Unemployment Rate by Region"
)

plt.xlabel("Region")

plt.ylabel(
    "Average Unemployment Rate (%)"
)

plt.tight_layout()

plt.show()


# -------------------------------
# 12. SEASONAL ANALYSIS
# -------------------------------

df["Month"] = df["Date"].dt.month

seasonal = (
    df.groupby("Month")[rate_column]
    .mean()
)

print("\n" + "=" * 60)
print("SEASONAL ANALYSIS")
print("=" * 60)

print(
    seasonal.round(2)
)


# -------------------------------
# 13. SEASONAL TREND GRAPH
# -------------------------------

month_names = [
    "Jan", "Feb", "Mar", "Apr",
    "May", "Jun", "Jul", "Aug",
    "Sep", "Oct", "Nov", "Dec"
]

plt.figure(figsize=(10, 5))

plt.plot(
    seasonal.index,
    seasonal.values,
    marker="o"
)

plt.title(
    "Seasonal Pattern of Unemployment"
)

plt.xlabel("Month")

plt.ylabel(
    "Average Unemployment Rate (%)"
)

plt.xticks(
    range(1, 13),
    month_names
)

plt.grid(True)

plt.tight_layout()

plt.show()


# -------------------------------
# 14. YEARLY ANALYSIS
# -------------------------------

df["Year"] = df["Date"].dt.year

yearly = (
    df.groupby("Year")[rate_column]
    .mean()
)

print("\n" + "=" * 60)
print("YEARLY AVERAGE UNEMPLOYMENT")
print("=" * 60)

print(
    yearly.round(2)
)


# -------------------------------
# 15. YEARLY TREND GRAPH
# -------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    yearly.index,
    yearly.values,
    marker="o"
)

plt.title(
    "Yearly Average Unemployment Rate"
)

plt.xlabel("Year")

plt.ylabel(
    "Average Unemployment Rate (%)"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# -------------------------------
# 16. HEATMAP
# -------------------------------

heatmap_data = df.pivot_table(
    values=rate_column,
    index="Year",
    columns="Month",
    aggfunc="mean"
)

plt.figure(figsize=(12, 6))

sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".1f"
)

plt.title(
    "Year-Month Unemployment Rate Heatmap"
)

plt.xlabel("Month")

plt.ylabel("Year")

plt.tight_layout()

plt.show()


# -------------------------------
# 17. HIGHEST AND LOWEST
# -------------------------------

highest = df.loc[
    df[rate_column].idxmax()
]

lowest = df.loc[
    df[rate_column].idxmin()
]

print("\n" + "=" * 60)
print("HIGHEST AND LOWEST UNEMPLOYMENT")
print("=" * 60)

print("\nHighest Unemployment:")

print(
    "Date:",
    highest["Date"].strftime("%Y-%m-%d")
)

print(
    "Rate:",
    round(highest[rate_column], 2),
    "%"
)

print("\nLowest Unemployment:")

print(
    "Date:",
    lowest["Date"].strftime("%Y-%m-%d")
)

print(
    "Rate:",
    round(lowest[rate_column], 2),
    "%"
)


# -------------------------------
# 18. KEY PATTERNS
# -------------------------------

highest_month = seasonal.idxmax()
lowest_month = seasonal.idxmin()

print("\n" + "=" * 60)
print("KEY PATTERNS")
print("=" * 60)

print(
    "\nHighest seasonal month:",
    month_names[highest_month - 1]
)

print(
    "Lowest seasonal month:",
    month_names[lowest_month - 1]
)

highest_region = region_average.idxmax()
lowest_region = region_average.idxmin()

print(
    "\nRegion with highest average unemployment:",
    highest_region
)

print(
    "Region with lowest average unemployment:",
    lowest_region
)


# -------------------------------
# 19. POLICY RECOMMENDATIONS
# -------------------------------

print("\n" + "=" * 60)
print("POLICY RECOMMENDATIONS")
print("=" * 60)

print("""
1. Create targeted employment programs for regions
   with high unemployment.

2. Increase skill-development and vocational training.

3. Provide support to businesses during economic crises.

4. Encourage investment in job-creating industries.

5. Develop emergency employment-support programs.

6. Monitor unemployment rates regularly to identify
   sudden increases.

7. Use unemployment trends to improve economic planning.
""")


# -------------------------------
# 20. FINAL CONCLUSION
# -------------------------------

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)

print("""
The analysis shows how unemployment changes over time.
The COVID-19 period shows a significant increase in
unemployment compared with normal periods.

Regional and seasonal analysis helps identify differences
in unemployment levels. These findings can help governments
and policymakers design employment programs, skill-training
initiatives and economic-support policies.
""")

print("=" * 60)
print("     ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 60)