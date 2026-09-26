# TASK 3: Car Price Prediction using Machine Learning
# Python | Pandas | Scikit-learn | Matplotlib

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 1. CREATE DATASET
# =========================================================

data = {
    "brand": [
        "Toyota", "Honda", "Hyundai", "Maruti", "Tata",
        "Ford", "Kia", "Toyota", "Honda", "Hyundai",
        "Maruti", "Tata", "Ford", "Kia", "Toyota",
        "Honda", "Hyundai", "Maruti", "Tata", "Kia"
    ],

    "horsepower": [
        105, 120, 115, 90, 110,
        125, 140, 150, 130, 100,
        85, 115, 135, 145, 160,
        125, 110, 95, 120, 150
    ],

    "mileage": [
        18.5, 16.2, 17.5, 20.1, 19.0,
        15.5, 14.8, 13.5, 15.8, 18.2,
        21.0, 18.5, 16.0, 14.5, 12.5,
        16.0, 17.8, 20.5, 17.0, 13.8
    ],

    "engine_size": [
        1.5, 1.6, 1.5, 1.2, 1.5,
        1.6, 2.0, 2.0, 1.8, 1.5,
        1.2, 1.5, 1.6, 2.0, 2.2,
        1.8, 1.5, 1.2, 1.6, 2.0
    ],

    "year": [
        2020, 2021, 2019, 2022, 2021,
        2019, 2022, 2023, 2020, 2021,
        2023, 2020, 2018, 2022, 2023,
        2021, 2020, 2022, 2019, 2023
    ],

    "brand_goodwill": [
        8.5, 8.8, 8.0, 7.8, 7.9,
        7.5, 8.3, 8.5, 8.8, 8.0,
        7.8, 7.9, 7.5, 8.3, 8.5,
        8.8, 8.0, 7.8, 7.9, 8.3
    ],

    "price": [
        850000, 1050000, 720000, 650000, 780000,
        900000, 1250000, 1450000, 1150000, 800000,
        700000, 820000, 850000, 1350000, 1750000,
        1200000, 850000, 720000, 750000, 1400000
    ]
}

df = pd.DataFrame(data)


# =========================================================
# 2. DISPLAY DATASET
# =========================================================

print("\n==============================")
print("CAR PRICE PREDICTION")
print("==============================")

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())


# =========================================================
# 3. SELECT FEATURES AND TARGET
# =========================================================

X = df[
    [
        "brand",
        "horsepower",
        "mileage",
        "engine_size",
        "year",
        "brand_goodwill"
    ]
]

y = df["price"]


# =========================================================
# 4. TRAIN-TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# =========================================================
# 5. DATA PREPROCESSING
# =========================================================

categorical_features = ["brand"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "brand_encoding",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# =========================================================
# 6. CREATE MACHINE LEARNING MODEL
# =========================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)


# =========================================================
# 7. TRAIN MODEL
# =========================================================

print("\nTraining model...")

model.fit(X_train, y_train)

print("Model training completed successfully.")


# =========================================================
# 8. MAKE PREDICTIONS
# =========================================================

y_pred = model.predict(X_test)


# =========================================================
# 9. MODEL EVALUATION
# =========================================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print(f"Mean Absolute Error  : ₹{mae:,.2f}")
print(f"Mean Squared Error   : {mse:,.2f}")
print(f"Root Mean Square Error: ₹{rmse:,.2f}")
print(f"R2 Score             : {r2:.2f}")


# =========================================================
# 10. ACTUAL VS PREDICTED VALUES
# =========================================================

comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred.round(2)
})

print("\nActual vs Predicted Prices:")
print(comparison)


# =========================================================
# 11. VISUALIZATION
# =========================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.8
)

plt.xlabel("Actual Car Price")
plt.ylabel("Predicted Car Price")
plt.title("Actual vs Predicted Car Prices")

plt.grid(True)
plt.tight_layout()

plt.show()


# =========================================================
# 12. PREDICT PRICE OF A NEW CAR
# =========================================================

new_car = pd.DataFrame({
    "brand": ["Toyota"],
    "horsepower": [130],
    "mileage": [17.0],
    "engine_size": [1.8],
    "year": [2023],
    "brand_goodwill": [8.5]
})

predicted_price = model.predict(new_car)

print("\n==============================")
print("NEW CAR PRICE PREDICTION")
print("==============================")

print("Car Details:")
print(new_car)

print(
    f"\nPredicted Car Price: ₹{predicted_price[0]:,.2f}"
)

print("\nProgram executed successfully.")